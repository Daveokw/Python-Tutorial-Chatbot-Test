# app.py
import os
import re
import pickle
import gdown
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dotenv import load_dotenv

# ------------------------
# Config / env
# ------------------------
load_dotenv()
GDRIVE_INDEX_FAISS_ID = os.getenv("GDRIVE_INDEX_FAISS_ID")
GDRIVE_INDEX_PKL_ID = os.getenv("GDRIVE_INDEX_PKL_ID")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")  # override to "google/flan-t5-small" if limited

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
        # ignore; load_index_and_chunks() will handle missing files
        pass

# ------------------------
# Cached resource loaders
# ------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_extractive_qa():
    # Fast extractive QA
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)

@st.cache_resource(show_spinner=False)
def load_gen_pipeline(model_name=HF_MODEL):
    # Load text2text generator; choose device based on availability
    # Use device=-1 for CPU, 0 for CUDA (if available)
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1
    return pipeline("text2text-generation", model=model_name, device=device)

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
# Cleaning & utilities
# ------------------------
BOILERPLATE_PATTERNS = [
    r"\bGet Certified\b", r"\bSign In\b", r"\bTryit Editor\b", r"\bSearch\b",
    r"\bSpaces\b", r"\bMenu\b", r"\bAbout\b", r"\bContact\b", r"\bSubscribe\b",
    r"W3Schools", r"©", r"Privacy", r"Terms", r"Home\b", r"Next\b", r"Prev\b",
    r"Advertisement", r"©\s*\d{4}"
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\r\n", "\n", s)
    lines = []
    for ln in (line.strip() for line in s.splitlines()):
        if not ln:
            continue
        # drop obvious nav/menu lines
        if len(ln) < 30 and (ln.isupper() or len(ln.split()) <= 3):
            continue
        if BOILERPLATE_RE.search(ln):
            continue
        lines.append(ln)
    out = "\n".join(lines)
    out = re.sub(r"\n{2,}", "\n\n", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()

# ------------------------
# Search + QA pipeline
# ------------------------
embed_model = load_embed_model()
extractive_qa = load_extractive_qa()
gen_pipeline = load_gen_pipeline()

def search_index(idx, chunks, question, k=8):
    qv = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = idx.search(qv, k)
    results = []
    for dist, idx_i in zip(D[0], I[0]):
        if idx_i < len(chunks):
            results.append((chunks[idx_i], float(dist)))
    return results

def extract_answers_from_hits(hits_with_dist, question, keep_top=6):
    candidates = []
    for h, dist in hits_with_dist:
        text = clean_text(h.get("text",""))
        if not text or len(text) < 60:
            continue
        try:
            res = extractive_qa(question=question, context=text)
            score = float(res.get("score", 0.0))
            ans = res.get("answer","").strip()
            if ans:
                candidates.append({
                    "answer": ans,
                    "score": score,
                    "source": h.get("url",""),
                    "text": text,
                    "dist": dist
                })
        except Exception:
            continue
    candidates.sort(key=lambda x: (-x["score"], x["dist"]))
    # dedupe
    unique = []
    seen = set()
    for c in candidates:
        key = c["answer"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
        if len(unique) >= keep_top:
            break
    return unique

def synthesize_answer_rich(question, candidates, max_len=320):
    """
    Build a rich prompt that:
    - uses candidates as facts
    - allows safe background facts (creator/release year) for programming languages
    - asks for structured output (definition, features, use-cases, example)
    """
    if not candidates:
        return None, []

    # Build short facts list from top candidates
    facts = []
    sources = []
    for c in candidates[:4]:
        a = c["answer"].strip()
        if len(a) < 8:
            continue
        facts.append(f"- {a} (source: {c.get('source') or 'site'})")
        if c.get("source"):
            sources.append(c.get("source"))

    if not facts:
        return None, sources

    facts_block = "\n".join(facts)

    prompt = (
        "You are a helpful assistant. Use ONLY the facts below to produce a friendly, concise, AI-style answer. "
        "You MAY include a small amount of generally-known safe background facts for programming languages (creator and initial release year) "
        "if they are common knowledge — but do NOT invent other facts. If the facts below do not answer the question, reply exactly: "
        "\"I couldn't find this in the site data.\".\n\n"
        "Produce a response structured as:\n"
        "1) One-line definition/summary.\n"
        "2) 'What makes it special?' — 3 bullets.\n"
        "3) 'What can you build?' — 3 bullets.\n"
        "4) A tiny example code block (if relevant) wrapped in ```python```\n"
        "5) One encouraging closing sentence.\n\n"
        f"Question: {question}\n\nFacts (do not invent beyond allowed background facts):\n{facts_block}\n\nAnswer:"
    )
    try:
        out = gen_pipeline(prompt, max_length=max_len, do_sample=False)
        text = out[0].get("generated_text","").strip()
        text = re.sub(r"\s{2,}", " ", text).strip()
        # get top up to 2 unique sources
        unique_sources = []
        for s in sources:
            if s not in unique_sources and len(unique_sources) < 2:
                unique_sources.append(s)
        return text, unique_sources
    except Exception:
        return None, []

# ------------------------
# UI (minimal, no download messages)
# ------------------------
st.set_page_config(page_title="W3Schools Python Chatbot", layout="centered")
st.title("🐍 W3Schools Python — Q&A")

# Ensure index (silent)
download_from_gdrive_if_missing()
INDEX, CHUNKS = load_index_and_chunks()

if INDEX is None or CHUNKS is None:
    st.warning("Index not available. Upload index files to Google Drive and set GDRIVE_INDEX_FAISS_ID and GDRIVE_INDEX_PKL_ID as secrets, or run the index builder locally.")
    st.text_input("Ask a question about W3Schools Python (index not loaded):")
    st.stop()

query = st.text_input("Ask a question about W3Schools Python:", value="", key="query_input")
ask = st.button("Ask")

if ask and query.strip():
    with st.spinner("Thinking..."):
        try:
            hits = search_index(INDEX, CHUNKS, query, k=10)
            candidates = extract_answers_from_hits(hits, query, keep_top=8)
            answer, top_sources = None, []
            if candidates:
                answer, top_sources = synthesize_answer_rich(query, candidates, max_len=360)
            else:
                # fallback: raw context from top chunks
                raw = " \n\n ".join([clean_text(h.get("text","")) for h, _ in hits if len(clean_text(h.get("text","")))>60][:4])
                if raw:
                    # create pseudo-candidate so generator has something to work from
                    pseudo = [{"answer": raw, "score": 0.01, "source": ""}]
                    answer, top_sources = synthesize_answer_rich(query, pseudo, max_len=360)
            if answer:
                low = answer.lower()
                if "couldn't find" in low or "could not find" in low or "i couldn't find" in low:
                    st.info("I couldn't find this in the site data.")
                else:
                    st.markdown("### 🧠 Answer")
                    st.markdown(answer)
                    # Compact sources line (optional)
                    if top_sources:
                        links = ", ".join(f"[{s}]({s})" for s in top_sources)
                        st.markdown(f"**Sources:** {links}")
            else:
                st.info("I couldn't find a confident answer in the indexed content.")
        except Exception:
            st.error("An error occurred while searching or answering. Please try again.")
