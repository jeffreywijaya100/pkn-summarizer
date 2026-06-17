import os
import re
import uuid
import traceback
import numpy as np
import torch

from fastapi import FastAPI, UploadFile, File, Request, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from text_struct import extract_pdf, build_structure, split_sentences

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

MODEL_NAME = "jeffreywijaya100/model_roberta_pkn_summarizer"
MAX_LEN = 96
BATCH_SIZE = 8

DEVICE = torch.device("cpu")  # safer for HF / Railway / Vercel-like env

app = FastAPI(title="PKN Summarizer (Production Safe)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# LOG MIDDLEWARE
# ─────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"[REQ] {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        print(f"[RES] {response.status_code}")
        return response
    except Exception:
        print(traceback.format_exc())
        return JSONResponse(
            {"error": "Internal server error"},
            status_code=500
        )

# ─────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(req, exc):
    return JSONResponse(
        {"error": str(exc.detail)},
        status_code=exc.status_code
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(req, exc):
    errors = "; ".join([f"{e['loc'][-1]}: {e['msg']}" for e in exc.errors()])
    return JSONResponse({"error": errors}, status_code=422)

@app.exception_handler(Exception)
async def general_exception_handler(req, exc):
    print(traceback.format_exc())
    return JSONResponse(
        {"error": str(exc)},
        status_code=500
    )

# ─────────────────────────────────────────────────────────────
# MODEL (LAZY LOADING - IMPORTANT)
# ─────────────────────────────────────────────────────────────

tokenizer = None
model = None
model_loaded = False

def load_model():
    global tokenizer, model, model_loaded

    if model_loaded:
        return

    try:
        print(f"[INFO] Loading model: {MODEL_NAME}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        model.to(DEVICE)
        model.eval()

        model_loaded = True
        print(f"[OK] Model loaded on {DEVICE}")

    except Exception as e:
        print("[ERROR] Failed to load model")
        print(traceback.format_exc())
        model_loaded = False

# ─────────────────────────────────────────────────────────────
# MEMORY CACHE (DOC STRUCTURE)
# ─────────────────────────────────────────────────────────────

DOC_CACHE = {}
DOC_ORDER = []
MAX_DOCS = 20

def cache_put(doc_id, nodes):
    DOC_CACHE[doc_id] = nodes
    DOC_ORDER.append(doc_id)

    while len(DOC_ORDER) > MAX_DOCS:
        old = DOC_ORDER.pop(0)
        DOC_CACHE.pop(old, None)

# ─────────────────────────────────────────────────────────────
# CONNECTOR CLEANING
# ─────────────────────────────────────────────────────────────

CONNECTOR_RE = re.compile(
    r'^(Selain\s+itu|Namun\s+demikian|Namun|Oleh\s+karena\s+itu|'
    r'Dengan\s+demikian|Akan\s+tetapi|Sementara\s+itu|Di\s+samping\s+itu|'
    r'Lebih\s+lanjut|Adapun|Oleh\s+sebab\s+itu|Hal\s+ini|Hal\s+tersebut|'
    r'Berkaitan\s+dengan\s+hal\s+tersebut|Lebih\s+jauh)\b',
    re.IGNORECASE
)

TRUNCATED_RE = re.compile(
    r'^(yang|dan|atau|juga|serta|karena|maka|agar|sehingga|bahwa|'
    r'dengan|untuk|dari|ke|di|pada|dalam|oleh|sebagai|apabila|jika)\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# MODEL INFERENCE
# ─────────────────────────────────────────────────────────────

def predict_scores(sentences):
    scores = []

    load_model()
    if model is None:
        return [0.0] * len(sentences)

    with torch.no_grad():
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i:i+BATCH_SIZE]

            enc = tokenizer(
                batch,
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt"
            ).to(DEVICE)

            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]

            scores.extend(probs.cpu().tolist())

    return scores

# ─────────────────────────────────────────────────────────────
# HEURISTIC SCORING
# ─────────────────────────────────────────────────────────────

def quality_score(sentence: str, idx: int) -> float:
    s = sentence.strip()
    score = 0.0

    if CONNECTOR_RE.match(s):
        score -= 0.25

    if TRUNCATED_RE.match(s):
        score -= 0.5

    if idx == 0:
        score += 0.15

    if re.search(r'\b(adalah|merupakan|ialah|yaitu|yakni)\b', s, re.IGNORECASE):
        score += 0.1

    if len(s.split()) < 6:
        score -= 0.3

    return score

# ─────────────────────────────────────────────────────────────
# SUMMARIZATION CORE
# ─────────────────────────────────────────────────────────────

def summarize_text(text: str, top_k: int = 2):
    sentences = split_sentences(text)

    if not sentences:
        return text[:300]

    bert_scores = predict_scores(sentences)

    final_scores = [
        bert_scores[i] + quality_score(s, i)
        for i, s in enumerate(sentences)
    ]

    top_k = min(top_k, len(sentences))
    idx = np.argsort(final_scores)[-top_k:]
    idx = sorted(idx)

    selected = [sentences[i] for i in idx]

    # fix bad opening sentence
    if selected and (CONNECTOR_RE.match(selected[0]) or TRUNCATED_RE.match(selected[0])):
        if sentences[0] not in selected:
            selected[0] = sentences[0]

    return " ".join(selected)

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "OK", "model_loaded": model_loaded}

@app.get("/status")
def status():
    return {
        "model_loaded": model_loaded,
        "model": MODEL_NAME,
        "device": str(DEVICE),
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()

    if file.filename.lower().endswith(".pdf"):
        text = extract_pdf(raw)
    else:
        text = raw.decode("utf-8", errors="ignore")

    if len(text.strip()) < 50:
        return JSONResponse({"error": "Text too short"}, status_code=400)

    chapters, nodes = build_structure(text)

    doc_id = uuid.uuid4().hex
    cache_put(doc_id, nodes)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chapters": chapters
    }

@app.post("/summarize")
async def summarize(payload: dict = Body(...)):
    load_model()

    if model is None:
        return JSONResponse(
            {"error": "Model failed to load"},
            status_code=503
        )

    doc_id = payload.get("doc_id")
    ids = payload.get("ids", [])
    top_k = int(payload.get("top_k", 2))

    nodes = DOC_CACHE.get(doc_id)

    if not nodes:
        return JSONResponse(
            {"error": "Document not found or expired"},
            status_code=404
        )

    results = {}

    for nid in ids:
        text = nodes.get(nid, "").strip()
        results[nid] = summarize_text(text, top_k) if text else ""

    return {"summaries": results}

# ─────────────────────────────────────────────────────────────
# LOCAL DEV ONLY
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
