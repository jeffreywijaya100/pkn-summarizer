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
import uvicorn

from text_struct import extract_pdf, build_structure, split_sentences

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR = "jeffreywijaya100/model_roberta_pkn_summarizer"
MAX_LEN  = 96
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="PKN Summarizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Error handlers ────────────────────────────────────────────────────────────
@app.middleware("http")
async def _log(request: Request, call_next):
    print(f"[REQ] {request.method} {request.url.path}")
    r = await call_next(request)
    print(f"[RES] {r.status_code}")
    return r

@app.exception_handler(StarletteHTTPException)
async def _http_err(req, exc):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

@app.exception_handler(RequestValidationError)
async def _val_err(req, exc):
    fields = [f"{e['loc'][-1]}: {e['msg']}" for e in exc.errors()]
    return JSONResponse({"error": "; ".join(fields)}, status_code=422)

@app.exception_handler(Exception)
async def _unhandled(req, exc):
    print(traceback.format_exc())
    return JSONResponse({"error": f"Server error: {exc}"}, status_code=500)

# ── Model ─────────────────────────────────────────────────────────────────────
tokenizer = None
model     = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    try:
        print(f"[INFO] Memuat tokenizer & model dari '{SAVE_DIR}' …")
        tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(DEVICE)
        model.eval()
        print(f"[OK] Model siap di {DEVICE}")
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        print(traceback.format_exc())

# ── Doc cache ─────────────────────────────────────────────────────────────────
DOC_CACHE = {}
DOC_ORDER = []
MAX_DOCS  = 20

def cache_put(doc_id, nodes):
    DOC_CACHE[doc_id] = nodes
    DOC_ORDER.append(doc_id)
    while len(DOC_ORDER) > MAX_DOCS:
        DOC_CACHE.pop(DOC_ORDER.pop(0), None)

# ── Inferensi RoBERTa ──────────────────────────────────────────────────────
def predict_sentence_scores(sentences, batch_size=32):
    scores = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = tokenizer(
                batch, truncation=True, max_length=MAX_LEN,
                padding=True, return_tensors="pt"
            ).to(DEVICE)
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.cpu().tolist())
    return scores

# ── Pola kalimat yang tidak layak jadi pembuka ringkasan ─────────────────────
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

def score_sentence_quality(sentence: str, idx: int) -> float:
    """Skor kualitas tambahan di luar skor RoBERTa."""
    score = 0.0
    s = sentence.strip()

    # Penalti: kalimat yang dimulai dengan kata penghubung tanpa konteks
    if CONNECTOR_RE.match(s):
        score -= 0.25

    # Penalti lebih berat: kemungkinan kalimat terpotong dari paragraf sebelumnya
    if TRUNCATED_RE.match(s):
        score -= 0.5

    # Bonus: kalimat pertama biasanya memperkenalkan topik sub bab
    if idx == 0:
        score += 0.15

    # Bonus: kalimat yang mengandung pola definisi → informatif dan berdiri sendiri
    if re.search(r'\b(?:adalah|merupakan|ialah|yaitu|yakni)\b', s, re.IGNORECASE):
        score += 0.1

    # Penalti: kalimat terlalu pendek (< 6 kata) → kurang informatif
    if len(s.split()) < 6:
        score -= 0.3

    return score

def summarize_text(text: str, top_k: int = 2) -> str:

    sentences = split_sentences(text)
    if not sentences:
        return text[:300].strip()

    # Skor RoBERTa
    bert_scores = predict_sentence_scores(sentences)

    # Gabung skor BERT + skor kualitas kalimat
    final_scores = [
        b + score_sentence_quality(s, i)
        for i, (s, b) in enumerate(zip(sentences, bert_scores))
    ]

    top_k = min(top_k, len(sentences))

    # Ambil indeks top-k, urutkan sesuai posisi asli agar kalimat mengalir natural
    indices = sorted(np.argsort(final_scores)[-top_k:])
    selected = [sentences[i] for i in indices]

    # Jika kalimat pertama yang terpilih masih menggantung (connector/terpotong),
    # ganti dengan kalimat pertama teks asli sebagai kalimat topik/pembuka
    if selected and (CONNECTOR_RE.match(selected[0]) or TRUNCATED_RE.match(selected[0])):
        if sentences[0] not in selected:
            selected[0] = sentences[0]

    return " ".join(selected)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse("index.html")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    if file.filename.lower().endswith(".pdf"):
        text = extract_pdf(raw)
    else:
        text = raw.decode("utf-8", errors="ignore")

    if len(text.strip()) < 50:
        return JSONResponse(
            {"error": "Teks terlalu pendek atau PDF gagal diekstrak."},
            status_code=400
        )

    chapters, nodes = build_structure(text)
    doc_id = uuid.uuid4().hex
    cache_put(doc_id, nodes)

    return JSONResponse({
        "doc_id": doc_id,
        "filename": file.filename,
        "chapters": chapters,
    })

@app.post("/summarize")
async def summarize(payload: dict = Body(...)):
    if model is None:
        return JSONResponse(
            {"error": "Model belum siap. Tunggu beberapa detik."},
            status_code=503
        )

    doc_id = payload.get("doc_id")
    ids    = payload.get("ids") or []
    top_k  = int(payload.get("top_k", 2))

    nodes = DOC_CACHE.get(doc_id)
    if nodes is None:
        return JSONResponse(
            {"error": "Dokumen tidak ditemukan atau sudah kedaluwarsa. Silakan analisis ulang."},
            status_code=404
        )

    summaries = {}
    for nid in ids:
        content = nodes.get(nid, "").strip()
        summaries[nid] = summarize_text(content, top_k=top_k) if content else ""

    return JSONResponse({"summaries": summaries})

@app.get("/status")
async def status():
    return {
        "model_loaded": model is not None,
        "fine_tuned": True,
        "model_dir": SAVE_DIR,
        "device": str(DEVICE),
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
