#!/usr/bin/env python3
"""
app.py - minimal Streamlit front-end for the tutorial index.

UI: title, single textbox, Ask button, concise answer display.
Behavior:
 - Loads data/index.faiss and data/index.pkl from ./data/ or downloads them from Google Drive if GDRIVE_* env vars set.
 - Retrieval: sentence-transformers + FAISS
 - Answering: extractive QA (distilbert) for a concise one-line reply
 - Optional: "Show expanded answer" button to generate a friendly multi-paragraph answer using a small HF generator.
"""
import os
import re
import pickle
import streamlit as st
import gdown
import faiss
import numpy as np
from dotenv import load_dotenv

# Transformers and sentence-transformers
from sentence_transformers import SentenceTransformer
from transformers import pipeline

load_dotenv()

GDRIVE_INDEX_FAISS_ID = os.getenv("GDRIVE_INDEX_FAISS_ID")
GDRIVE_INDEX_PKL_ID = os.getenv("GDRIVE_INDEX_PKL_ID")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")  # small by default (change if you have resources)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")

# ---------------- helper: download index from Drive if env provided (quiet) ----------------
def download_from_gdrive_if_missing():
    try:
        if GDRIVE_INDEX_FAISS_ID and not os.path.exists(INDEX_FAISS):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_INDEX_FAISS_ID}", INDEX_FAISS, quiet=True)
        if GDRIVE_INDEX_PKL_ID and not os.path.exists(INDEX_PKL):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_INDEX_PKL_ID}", INDEX_PKL, quiet=True)
    except Exception:
        pass

# ---------------- cached loaders ----------------
@st.cache_resource(show_spinner=False)
def load_embed_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_extractive_qa():
    # CPU-friendly extractive QA
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)

@st.cache_resource(show_spinner=False)
def load_generator(model_name=HF_MODEL):
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1
    try:
        return pipeline("text2text-generation", model=model_name, device=device)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_index_and_chunks():
    try:
        if not (os.path.exists(INDEX_FAISS) and os.path.exists(INDEX_PKL)):
            return None, None
        idx = faiss.read_index(INDEX_FAISS)
        with open(INDEX_PKL, "rb") as f:
            chunks = pickle.load(f)
        return idx, chunks
    except Exception:
        return None, None

# ---------------- cleaning helper ----------------
BOILERPLATE_PATTERNS = ["W3Schools", "©", "Privacy", "Terms", "Search", "Get Certified", "Sign In", "Menu"]
BOILERPLATE_RE = re.compile("|".join([re.escape(p) for p in BOILERPLATE_PATTERNS]), re.IGNORECASE)

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\r\n", "\n", s)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    out = "\n".join(lines)
    out = BOILERPLATE_RE.sub("", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()

# ---------------- retrieval & QA ----------------
embed_model = load_embed_model()
extractive_qa = load_extractive_qa()
generator = load_generator()

def search_index(idx, chunks, question, k=6):
    qv = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = idx.search(qv, k)
    results = []
    for dist, idx_i in zip(D[0], I[0]):
        if idx_i < len(chunks):
            results.append((chunks[idx_i], float(dist)))
    return results

def normalize_answer(a: str) -> str:
    s = re.sub(r'\s+', ' ', a.strip().lower())
    s = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', s)
    return s

def pick_best_candidate(candidates):
    if not candidates:
        return None
    best = {}
    for c in candidates:
        key = normalize_answer(c.get("answer",""))
        if not key:
            continue
        prev = best.get(key)
        if not prev:
            best[key] = c
        else:
            if c.get("score",0) > prev.get("score",0) + 1e-6:
                best[key] = c
    chosen = sorted(best.values(), key=lambda x: -x.get("score",0))
    return chosen[0] if chosen else None

def extract_candidates(hits_with_dist, question, keep_top=8):
    cand = []
    for h, dist in hits_with_dist:
        text = clean_text(h.get("text",""))
        if not text or len(text) < 60:
            continue
        try:
            res = extractive_qa(question=question, context=text)
            score = float(res.get("score", 0.0))
            ans = (res.get("answer") or "").strip()
            if ans:
                cand.append({
                    "answer": ans,
                    "score": score,
                    "source": h.get("url",""),
                    "text": text,
                    "dist": dist
                })
        except Exception:
            continue
    cand = sorted(cand, key=lambda x: -x.get("score",0))[:keep_top]
    return cand

def get_concise_answer(question, candidates):
    best = pick_best_candidate(candidates)
    if best:
        one = best.get("answer","").strip()
        one = re.sub(r'(\b\w+\b)(?:\s+\1\b){2,}', r'\1', one)  # simple dedupe
        if not re.search(r'[.!?]$', one):
            one = one + '.'
        return one, best.get("source","")
    return None, None

def synthesize_expanded(question, candidates, max_len=280):
    if not candidates or generator is None:
        return None, []
    facts = []
    sources = []
    used = set()
    for c in candidates[:6]:
        a = c.get("answer","").strip()
        key = normalize_answer(a)
        if not key or key in used:
            continue
        used.add(key)
        facts.append("- " + a)
        s = c.get("source")
        if s and s not in sources:
            sources.append(s)
    if not facts:
        return None, sources[:2]
    prompt = (
        "You are a helpful assistant. Using ONLY the facts below, write a concise friendly answer: 1) one-line definition; "
        "2) three short bullets 'what makes it special'; 3) three short bullets 'what you can build'; 4) a tiny python example (if relevant).\n\n"
        f"Question: {question}\n\nFacts:\n" + "\n".join(facts) + "\n\nAnswer:"
    )
    try:
        out = generator(prompt, max_length=max_len, do_sample=False)
        text = out[0].get("generated_text","").strip()
        text = re.sub(r'\s{2,}', ' ', text)
        return text, sources[:2]
    except Exception:
        return None, sources[:2]

# ---------------- Streamlit UI (minimal) ----------------
st.set_page_config(page_title="Python Tutorial Chatbot", layout="centered")
st.title("🐍 Python Tutorial Chatbot")

# try to download index from Drive if IDs provided
download_from_gdrive_if_missing()
INDEX, CHUNKS = load_index_and_chunks()

if INDEX is None or CHUNKS is None:
    st.error("Index files missing. Run embed_index.py locally, upload data/index.faiss and data/index.pkl, or set GDRIVE_INDEX_FAISS_ID & GDRIVE_INDEX_PKL_ID.")
    st.stop()

q = st.text_input("Ask a question about the Python tutorial:", "")
if st.button("Ask") and q.strip():
    with st.spinner("Searching..."):
        hits = search_index(INDEX, CHUNKS, q, k=8)
        candidates = extract_candidates(hits, q, keep_top=8)
        concise, src = get_concise_answer(q, candidates)
        if concise:
            st.markdown("### Answer")
            st.write(concise)
            if src:
                st.markdown(f"**Source:** [{src}]({src})")
            # expanded content on demand
            if st.button("Show expanded answer"):
                expanded, sources = synthesize_expanded(q, candidates)
                if expanded:
                    st.markdown("### Expanded Answer")
                    st.write(expanded)
                    if sources:
                        st.markdown("**Sources:** " + ", ".join(f"[{s}]({s})" for s in sources))
                else:
                    st.info("No expanded answer available.")
        else:
            # fallback: try generator on summarized chunks
            st.info("No confident concise answer found in the index.")
