#!/usr/bin/env python3
"""
app.py - Streamlit front-end (improved concise-definition extraction)

UI: single textbox, Ask button, concise answer display.
Behavior:
 - Loads data/index.faiss and data/index.pkl from ./data/ or downloads them from Google Drive if GDRIVE_* env vars set.
 - Retrieval: sentence-transformers + FAISS
 - Answering: prefer a definition sentence found in top chunks; fallback to extractive QA; then summarizer; then generator.
 - Optional: "Show expanded answer" for a stronger AI-style response.
"""
import os
import re
import pickle
import streamlit as st
import gdown
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline

load_dotenv()

GDRIVE_INDEX_FAISS_ID = os.getenv("GDRIVE_INDEX_FAISS_ID")
GDRIVE_INDEX_PKL_ID = os.getenv("GDRIVE_INDEX_PKL_ID")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")  # change if you have more resources

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")

# ---------------- helpers to download index quietly ----------------
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
    # CPU friendly
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)

@st.cache_resource(show_spinner=False)
def load_summarizer():
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    except Exception:
        return None

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

# ---------------- text cleaning ----------------
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
summarizer = load_summarizer()
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
    unique = sorted(best.values(), key=lambda x: -x.get("score",0))
    return unique[0] if unique else None

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

# ---------------- definition-finding heuristic ----------------
SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

def find_definition_in_hits(hits_with_dist, keyword="python", max_pages=6):
    """
    Search top retrieved chunks for a sentence that looks like a definition of the keyword.
    Returns the first matching sentence (cleaned) or None.
    """
    key = keyword.lower()
    # common definition verbs/phrases to look for
    patterns = [
        r'\b' + re.escape(keyword) + r'\b.*?\b(is|was|refers to|is an|is a|are|means|provides|lets|allows)\b',
        r'\bthe ' + re.escape(keyword) + r'\b.*?\b(is|was|refers to|is an|is a|are|means)\b'
    ]
    compiled = [re.compile(pat, re.I) for pat in patterns]
    # examine top hits (use the provided ordering)
    for h, dist in hits_with_dist[:max_pages]:
        text = clean_text(h.get("text",""))
        if not text:
            continue
        # split into sentences (simple)
        sents = SENT_SPLIT_RE.split(text)
        # check each sentence for keyword + definition pattern
        for s in sents:
            if key not in s.lower():
                continue
            s_stripped = s.strip()
            # minimal length to avoid navigation-like lines
            if len(s_stripped) < 30:
                continue
            for cre in compiled:
                if cre.search(s_stripped):
                    # ensure it ends with a period
                    if not re.search(r'[.!?]$', s_stripped):
                        s_stripped = s_stripped + '.'
                    # cleanup duplicates & whitespace
                    s_stripped = re.sub(r'\s{2,}', ' ', s_stripped).strip()
                    return s_stripped
    # fallback: return first sentence in top hit that contains keyword
    for h, dist in hits_with_dist[:max_pages]:
        text = clean_text(h.get("text",""))
        if not text:
            continue
        for s in SENT_SPLIT_RE.split(text):
            if key in s.lower() and len(s.strip()) >= 25:
                s_stripped = s.strip()
                if not re.search(r'[.!?]$', s_stripped):
                    s_stripped += '.'
                return re.sub(r'\s{2,}', ' ', s_stripped).strip()
    return None

# ---------------- concise / expanded answer logic ----------------
def get_concise_answer_from_flow(question, hits, candidates):
    """
    Preference order:
    1) Definition sentence found in top chunks
    2) Best extractive candidate (deduped)
    3) Summarizer over top chunks
    4) Generator producing a single-line answer (last resort)
    """
    # 1) try to find a definition sentence in the retrieved chunks
    def_sent = find_definition_in_hits(hits, keyword="python", max_pages=6)
    if def_sent:
        return def_sent, None

    # 2) try pick_best_candidate
    best = pick_best_candidate(candidates)
    if best:
        one = best.get("answer","").strip()
        # remove repeated words and trailing noise
        one = re.sub(r'(\b\w+\b)(?:\s+\1\b){2,}', r'\1', one)
        if not re.search(r'[.!?]$', one):
            one = one + '.'
        return one, best.get("source","")

    # 3) summarizer fallback over top chunks
    if summarizer and hits:
        texts = []
        for h, d in hits[:3]:
            t = clean_text(h.get("text",""))
            if t:
                texts.append(t)
        joined = "\n\n".join(texts)
        if joined and len(joined) > 100:
            try:
                out = summarizer(joined, max_length=60, min_length=15, do_sample=False)
                summ = out[0].get("summary_text","").strip()
                if summ:
                    if not re.search(r'[.!?]$', summ):
                        summ += '.'
                    return summ, None
            except Exception:
                pass

    # 4) generator fallback (try to create single-line answer using facts)
    if generator and candidates:
        facts = []
        used = set()
        for c in candidates[:5]:
            a = c.get("answer","").strip()
            k = normalize_answer(a)
            if not k or k in used:
                continue
            used.add(k)
            facts.append("- " + a)
        if facts:
            prompt = (
                "Using only the facts below, write one short definition (one sentence) for 'Python'. "
                "If the facts are insufficient, be concise and say 'I couldn't find this in the site data.'\n\n"
                "Facts:\n" + "\n".join(facts) + "\n\nDefinition:"
            )
            try:
                out = generator(prompt, max_length=60, do_sample=False)
                gen = out[0].get("generated_text","").strip()
                gen = re.sub(r'\s{2,}', ' ', gen)
                if gen:
                    # ensure it's short single-line
                    first_line = gen.splitlines()[0].strip()
                    if not re.search(r'[.!?]$', first_line):
                        first_line += '.'
                    return first_line, None
            except Exception:
                pass

    return None, None

def synthesize_expanded(question, candidates):
    """Generate the expanded friendly answer (same as previous behavior)."""
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
        return None, []
    prompt = (
        "You are a helpful assistant. Using ONLY the facts below, write a concise friendly answer structured as: "
        "1) One-line definition; 2) three short bullets 'what makes it special'; 3) three short bullets 'what you can build'; "
        "4) a tiny python example (if relevant).\n\n"
        f"Question: {question}\n\nFacts:\n" + "\n".join(facts) + "\n\nAnswer:"
    )
    try:
        out = generator(prompt, max_length=280, do_sample=False)
        text = out[0].get("generated_text","").strip()
        text = re.sub(r'\s{2,}', ' ', text)
        return text, sources[:2]
    except Exception:
        return None, sources[:2]

# ---------------- Streamlit UI (minimal) ----------------
st.set_page_config(page_title="Python Tutorial Chatbot", layout="centered")
st.title("🐍 Python Tutorial Chatbot")

# try to download index if env provided
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
        concise, src = get_concise_answer_from_flow(q, hits, candidates)
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
            st.info("I couldn't find a confident answer in the indexed content.")
