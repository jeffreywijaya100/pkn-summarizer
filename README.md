# PKN Summarizer — Web-Based Automatic Summarization for Civics (PKN) Textbooks

A web application for automatic **extractive text summarization** of Indonesian Civics (Pendidikan Kewarganegaraan / PKN) textbooks. Users upload a PKN textbook as a PDF, the system structures its content by chapter and sub-chapter, and generates summaries for the sections the user selects.

This project is the implementation of an undergraduate thesis (skripsi) in Data Science titled **"Development of a Web-Based Application for Text Summarization of Civics (PKN) Textbooks."**

---

## Overview

PKN material tends to be long, dense, and conceptual, making it hard for students to identify the key points in each sub-chapter. This application reframes summarization as **binary sentence classification**: each sentence is classified as summary-worthy (class 1) or not (class 0). At inference time, the class-1 probability is used as a **relevance score** to rank sentences, and the system selects the top-scoring sentences (*top-k*) as the extractive summary.

The extractive approach was chosen because its output is guaranteed to be verbatim from the source, meaning it **cannot hallucinate** — an important property in an educational context.

## Features

- Upload PKN textbooks as PDFs, with automatic removal of copyright/catalog pages (KDT), repeating running headers/footers, and page numbers.
- Automatic document-structure detection (chapter → sub-chapter) so users can choose exactly which sections to summarize.
- Per-sub-chapter summarization with adjustable summary length (`top_k` parameter).
- Sentence-quality heuristics that avoid dangling opening sentences (connectors / truncated clauses) and favor definition sentences.
- Lightweight web interface with no frontend framework.

## System Architecture

```
┌─────────────┐   PDF     ┌──────────────────────────────┐
│  index.html │ ────────► │  FastAPI (app.py)            │
│  (frontend) │           │                              │
│             │ ◄──────── │  text_struct.py              │
└─────────────┘ structure │   • extract_pdf (PyMuPDF)    │
      │                   │   • build_structure          │
      │  select sections  │   • split_sentences          │
      │  + top_k          │                              │
      └─────────────────► │  RoBERTa (fine-tuned)        │
                          │   • softmax → relevance score│
                          │   • top-k ranking + heuristic│
                          └──────────────┬───────────────┘
                                         │
                                  Hugging Face Hub
                       jeffreywijaya100/model_roberta_pkn_summarizer
```

Workflow:

1. **`POST /analyze`** — Upload a PDF. The text is extracted, cleaned, and structured into a chapter/sub-chapter tree. The structure is stored in a document cache (LRU, max 20 documents) and returned together with a `doc_id`.
2. **`POST /summarize`** — Send the `doc_id`, a list of selected sub-chapter `ids`, and `top_k`. Each sub-chapter is split into sentences, scored by the model, and the top sentences are reassembled in their original order.
3. **`GET /status`** — Model readiness status.

## Model & Approach

The model is trained as a sentence classifier but deployed as a continuous scorer: `softmax(logits)[:, 1]` becomes the sentence ranking score. The model score is then combined with a heuristic sentence-quality score before top-k selection.

The research compares five encoder-only Transformer models under a binary sentence-classification framing:

| Model | Checkpoint |
|---|---|
| BERT-Base | `bert-base-uncased` |
| DistilBERT | `distilbert-base-uncased` |
| RoBERTa | `flax-community/indonesian-roberta-base` |
| IndoBERT | `indobenchmark/indobert-base-p1` |
| ELECTRA | `google/electra-base-discriminator` |

Labeling follows a *distant supervision* principle: labels are derived by verbatim sentence matching against reference summaries (*oracle labeling*).

### Results

RoBERTa (`flax-community/indonesian-roberta-base`) was selected as the best-performing model and is the one deployed, based on the highest ROUGE scores in end-to-end evaluation:

| Metric | Score |
|---|---|
| ROUGE-1 | 0.556 |
| ROUGE-2 | 0.463 |
| ROUGE-L | 0.487 |

Because the system operates as a *relevance ranker* (top-k selection) rather than a threshold-based binary classifier, ROUGE is used as the primary metric for measuring final summary quality.

## Dataset

The dataset was built from **17 PKN textbooks** in PDF format spanning three curricula (Merdeka, 2013, and KTSP). After processing it yields **307 sub-chapters** (`dataset_pkn_fix.csv`) containing roughly **25,286 sentences**. Each row holds the source material and its reference summary. Splitting is done at the document level (*document-level split*) to prevent data leakage caused by intra-document sentence correlation.

`dataset_pkn_fix.csv` columns: `id_data`, `id_buku`, `jenjang`, `kelas`, `kurikulum`, `bab`, `judul_bab`, `sub_bab`, `judul_sub_bab`, `teks_materi`, `rangkuman_materi`.

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Model & NLP:** PyTorch, Hugging Face Transformers
- **PDF extraction:** PyMuPDF (`fitz`)
- **Frontend:** HTML + JavaScript (vanilla)
- **Model hosting:** Hugging Face Hub
- **Deployment:** Railway

## Running Locally

Prerequisites: Python 3.10+.

```bash
# 1. Clone the repository
git clone https://github.com/<username>/<repo>.git
cd <repo>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
python app.py
```

The server runs at `http://localhost:8000`. The model is downloaded automatically from the Hugging Face Hub at startup.

## Repository Structure

```
.
├── app.py                # FastAPI backend + inference pipeline
├── text_struct.py        # PDF extraction & chapter/sub-chapter detection
├── index.html            # Web interface
├── requirements.txt      # Python dependencies
├── dataset_pkn_fix.csv   # Dataset (307 sub-chapters)
├── config.json           # Model configuration
├── tokenizer.json        # Tokenizer
└── notebooks/            # Training notebooks for the five models
    ├── PKN_with_BERT-Base.ipynb
    ├── PKN_with_distillBERT.ipynb
    ├── PKN_with_RoBERTa.ipynb
    ├── PKN_with_ELECTRA.ipynb
    └── PKN_with_IndoBERT.ipynb
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the web interface |
| `POST` | `/analyze` | Upload a PDF, return chapter/sub-chapter structure + `doc_id` |
| `POST` | `/summarize` | Summarize selected sub-chapters (`doc_id`, `ids`, `top_k`) |
| `GET` | `/status` | Model readiness status |

## Notes

This repository is part of a thesis and focuses on system development rather than experiments on improving student learning outcomes. The deployed model is a RoBERTa checkpoint fine-tuned for sentence classification on the PKN domain.
