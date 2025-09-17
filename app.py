# app.py
import os
import pickle
import gdown
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv
import re

# ------------------------
# Config / env
# ------------------------
load_dotenv()
GDRIVE_INDEX_FAISS_ID = os.getenv("GDRIVE_INDEX_FAISS_ID")
GDRIVE_INDEX_PKL_ID = os.getenv("GDRIVE_INDEX_PKL_ID")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")

# ------------------------
# Helpers: download index from Google Drive (silent)
# ------------------------
def download_from_gdrive_if_missing():
    try:
        if GDRIVE_INDEX_FAISS_ID and not os.path.exists(INDEX_FAISS):
            url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_FAISS_ID}"
            gdown.download(url, INDEX_FAISS, quiet=True)
        if GDRIVE_INDEX_PKL_ID and not os.path.exists(INDEX_PKL):
            url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_PKL_ID}"
            gdown.download(url, INDEX_PKL, quiet=True)
    except Exception:
        pass

# ------------------------
# Cached resource loaders
# ------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_extractive_qa():
    # Fast extractive QA for per-chunk answers
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@st.cache_resource(show_spinner=False)
def load_gen_pipeline():
    # Generative model to synthesize final answer
    return pipeline("text2text-generation", model="google/flan-t5-small")

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

# ------------------------
# Text cleaning & utilities
# ------------------------
# common site/menu boilerplate terms to drop
BOILERPLATE_PATTERNS = [
    r"\bGet Certified\b", r"\bSign In\b", r"\bTryit Editor\b", r"\bSearch\b",
    r"\bSpaces\b", r"\bMenu\b", r"\bAbout\b", r"\bContact\b", r"\bSubscribe\b",
    r"W3Schools", r"©", r"Privacy", r"Terms", r"Home\b", r"Next\b", r"Prev\b",
    r"Advertisement", r"©\s*\d{4}"
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

def clean_text(s: str) -> str:
    """Remove obvious navigation/boilerplate and short junk lines."""
    if not s:
        return ""
    # remove repeated whitespace
    s = re.sub(r"\r\n", "\n", s)
    # drop lines that look like menus or are extremely short
    lines = []
    for ln in (line.strip() for line in s.splitlines()):
        if not ln:
            continue
        # drop short lines that are likely nav
        if len(ln) < 30 and (ln.isupper() or len(ln.split()) <= 3):
            continue
        # drop lines containing boilerplate keywords
        if BOILERPLATE_RE.search(ln):
            continue
        lines.append(ln)
    # join and collapse repeated whitespace
    out = "\n".join(lines)
    out = re.sub(r"\n{2,}", "\n\n", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()

# ------------------------
# Search & QA flow
# ------------------------
embed_model = load_embed_model()
extractive_qa = load_extractive_qa()
gen_pipeline = load_gen_pipeline()

def search_index(idx, chunks, question, k=6):
    """Return list of (chunk_dict, distance) pairs for top-k"""
    qv = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = idx.search(qv, k)
    results = []
    for dist, idx_i in zip(D[0], I[0]):
        if idx_i < len(chunks):
            results.append((chunks[idx_i], float(dist)))
    return results

def extract_answers_from_hits(hits_with_dist, question, min_score=0.10):
    """
    Run extractive QA on each hit. Returns list of dicts:
    {'answer':..., 'score':..., 'source': url, 'text': cleaned_text, 'dist': ...}
    Keep only candidates with score >= min_score (or keep top N).
    """
    candidates = []
    for h, dist in hits_with_dist:
        text = clean_text(h.get("text",""))
        if not text or len(text) < 50:
            continue
        try:
            res = extractive_qa(question=question, context=text)
            # pipeline returns {'score', 'start', 'end', 'answer'}
            score = float(res.get("score", 0.0))
            ans = res.get("answer","").strip()
            if ans and score >= 0.01:  # keep even low for later filtering
                candidates.append({
                    "answer": ans,
                    "score": score,
                    "source": h.get("url",""),
                    "text": text,
                    "dist": dist
                })
        except Exception:
            continue
    # sort by score desc then distance asc
    candidates.sort(key=lambda x: (-x["score"], x["dist"]))
    # optionally filter by score threshold or keep top 5
    return candidates[:6]

def synthesize_answer(question, candidates, max_len=180):
    """
    Given candidate extractive answers, produce one fluent answer using generative model.
    If candidates empty, return None.
    """
    if not candidates:
        return None

    # Keep the top 3 unique answers (by text)
    seen = set()
    top = []
    for c in candidates:
        a = c["answer"]
        if a.lower() in seen: 
            continue
        seen.add(a.lower())
        top.append(c)
        if len(top) >= 3:
            break

    # Build synthesis prompt: show candidate answers + short sources
    parts = []
    for i, c in enumerate(top, start=1):
        src = c.get("source") or "source"
        parts.append(f"Candidate {i}: {c['answer']} (source: {src})")
    candidates_block = "\n".join(parts)

    prompt = (
        "You are an assistant that must answer the user's question using ONLY the candidate answers below, "
        "which were extracted from the website. Combine and rewrite them into one clear, concise, AI-style answer. "
        "If the candidates contradict each other, synthesize the most likely correct information. "
        "Do NOT add facts not present in the candidates. If the candidates don't contain an answer, reply: "
        "\"I couldn't find this in the site data.\".\n\n"
        f"Question: {question}\n\nCandidates:\n{candidates_block}\n\nAnswer:"
    )

    try:
        out = gen_pipeline(prompt, max_length=max_len, do_sample=False)
        text = out[0].get("generated_text","").strip()
        # Post-clean: if the generator repeated candidate verbatim, condense whitespace
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text
    except Exception:
        return None

# ------------------------
# App UI (minimal)
# ------------------------
st.set_page_config(page_title="W3Schools Python Chatbot", layout="centered")
st.title("🐍 W3Schools Python — Q&A")

# Silent download + load index
download_from_gdrive_if_missing()
INDEX, CHUNKS = load_index_and_chunks()

if INDEX is None or CHUNKS is None:
    st.warning("Index not available. Please upload index files to Google Drive and set GDRIVE_INDEX_FAISS_ID and GDRIVE_INDEX_PKL_ID as secrets, or run the index builder locally.")
    st.text_input("Ask a question about W3Schools Python (index not loaded):")
    st.stop()

query = st.text_input("Ask a question about W3Schools Python:", value="", key="query_input")
ask = st.button("Ask")

if ask and query.strip():
    with st.spinner("Thinking..."):
        try:
            hits = search_index(INDEX, CHUNKS, query, k=8)  # retrieve more to improve candidate pool
            candidates = extract_answers_from_hits(hits, query, min_score=0.05)
            # If candidates are empty or low-confidence, we can still attempt to synthesize from raw chunks
            if not candidates:
                # fallback: build small context from top hits and run gen directly
                raw_context = " \n\n ".join([clean_text(h.get("text","")) for h, _ in hits if len(clean_text(h.get("text","")))>50][:4])
                answer = synthesize_answer(query, [{"answer": raw_context, "score": 0.01, "source": ""}]) if raw_context else None
            else:
                answer = synthesize_answer(query, candidates)
            if answer:
                # if model returns the explicit fallback phrase, show it exactly
                low = answer.lower()
                if "couldn't find" in low or "could not find" in low or "i couldn't find" in low:
                    st.info("I couldn't find this in the site data.")
                else:
                    st.markdown("### 🧠 Answer")
                    st.write(answer)
            else:
                st.info("I couldn't find a confident answer in the indexed content.")
        except Exception:
            st.error("An error occurred while searching or answering. Please try again.")
