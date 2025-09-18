#!/usr/bin/env python3
"""
app.py

Minimal Streamlit UI. Improved definition detection:
 - maps short queries (e.g., 'oop') to expanded phrase ('object oriented programming')
 - prefers sentences that explicitly define the term (e.g., 'Object-oriented programming is...')
 - removes encoding garbage (Â etc.)
 - falls back to extractive QA, summarizer, then generator
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

# Optional Drive IDs (set in Streamlit Cloud or .env)
GDRIVE_INDEX_FAISS_ID = os.getenv("GDRIVE_INDEX_FAISS_ID")
GDRIVE_INDEX_PKL_ID = os.getenv("GDRIVE_INDEX_PKL_ID")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")

# ------------- helper to download index quietly -------------
def download_from_gdrive_if_missing():
    try:
        if GDRIVE_INDEX_FAISS_ID and not os.path.exists(INDEX_FAISS):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_INDEX_FAISS_ID}", INDEX_FAISS, quiet=True)
        if GDRIVE_INDEX_PKL_ID and not os.path.exists(INDEX_PKL):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_INDEX_PKL_ID}", INDEX_PKL, quiet=True)
    except Exception:
        pass

# ------------- cached loaders -------------
@st.cache_resource(show_spinner=False)
def load_embed_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_extractive_qa():
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

# ------------- cleaning utilities -------------
# remove weird encoding artifacts and boilerplate tokens
def fix_encoding(s: str) -> str:
    if not s:
        return ""
    # replace garbage characters like Â
    s = s.replace("Â", " ")
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

BOILERPLATE_PATTERNS = ["©", "Privacy", "Terms", "Search", "Menu"]
BOILERPLATE_RE = re.compile("|".join([re.escape(p) for p in BOILERPLATE_PATTERNS]), re.IGNORECASE)

def clean_text(s: str) -> str:
    s = fix_encoding(s)
    if not s:
        return ""
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    out = "\n".join(lines)
    out = BOILERPLATE_RE.sub("", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()

# ------------- retrieval & QA -------------
embed_model = load_embed_model()
extractive_qa = load_extractive_qa()
summarizer = load_summarizer()
generator = load_generator()

def search_index(idx, chunks, question, k=8):
    qv = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = idx.search(qv, k)
    hits = []
    for dist, idx_i in zip(D[0], I[0]):
        if idx_i < len(chunks):
            hits.append((chunks[idx_i], float(dist)))
    return hits

def extractive_candidates(hits, question, keep_top=8):
    cand = []
    for h, dist in hits:
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
    cand = sorted(cand, key=lambda x: (-x["score"], x["dist"]))[:keep_top]
    return cand

# ------------- smarter definition-finder -------------
SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

# map short user terms to likely canonical phrase to look for
CANONICAL_MAP = {
    "oop": "object oriented programming",
    "object oriented programming": "object oriented programming",
    "object-oriented": "object oriented programming",
    "python": "python",
    "list": "list",
    "dictionary": "dictionary"
    # add more if you like
}

def canonicalize_keyword(q: str) -> str:
    q0 = q.lower().strip()
    q0 = re.sub(r'[^a-z0-9\s\-]', ' ', q0)
    q0 = re.sub(r'\s+', ' ', q0).strip()
    return CANONICAL_MAP.get(q0, q0)

def find_definition(hits, user_query, top_k_hits=8):
    """
    Attempt to find a clean one-sentence definition.
    Strategy:
      1) canonicalize query (e.g., 'oop' -> 'object oriented programming')
      2) search top hits for sentences that contain the canonical phrase plus 'is|was|refers to|means|is a|are' etc.
      3) prefer sentences from lower-distance hits
      4) fallback: any sentence containing canonical phrase with length >= 30
    """
    k = canonicalize_keyword(user_query)
    if not k:
        return None
    # build regexes
    verbs = r"(?:\b(is|was|are|refers to|means|provides|represents|describes|allows|supports)\b)"
    # permissive pattern: phrase ... is/was ... (captures definitions)
    pat1 = re.compile(rf".{{10,}}\b{re.escape(k)}\b.*?{verbs}.*?[\.!\?]", re.I)
    pat2 = re.compile(rf".{{10,}}\b{re.escape(k)}\b.*?[\.!\?]", re.I)
    # iterate hits prioritized by distance (smaller better)
    hits_sorted = sorted(hits, key=lambda x: x[1])[:top_k_hits]
    for h, dist in hits_sorted:
        text = clean_text(h.get("text",""))
        if not text:
            continue
        # split sentences
        sents = SENT_SPLIT_RE.split(text)
        for s in sents:
            s_clean = s.strip()
            if len(s_clean) < 30:
                continue
            if pat1.search(s_clean):
                # returned sentence
                one = s_clean
                if not re.search(r'[.!?]$', one):
                    one += '.'
                one = re.sub(r'\s{2,}', ' ', one).strip()
                return one, h.get("url","")
    # fallback: any sentence with the canonical phrase
    for h, dist in hits_sorted:
        text = clean_text(h.get("text",""))
        if not text:
            continue
        sents = SENT_SPLIT_RE.split(text)
        for s in sents:
            if k in s.lower() and len(s.strip()) >= 30:
                one = s.strip()
                if not re.search(r'[.!?]$', one):
                    one += '.'
                return re.sub(r'\s{2,}', ' ', one).strip(), h.get("url","")
    return None, None

# ------------- concise/expanded flows -------------
def get_concise_answer(question, hits, candidates):
    # 1) try definition finder
    def_sent, src = find_definition(hits, question, top_k_hits=8)
    if def_sent:
        return def_sent, src

    # 2) best extractive candidate
    if candidates:
        best = candidates[0]
        one = best.get("answer","").strip()
        one = re.sub(r'(\b\w+\b)(?:\s+\1\b){2,}', r'\1', one)
        if not re.search(r'[.!?]$', one):
            one += '.'
        return one, best.get("source","")

    # 3) summarizer fallback
    if summarizer and hits:
        texts = []
        for h, d in hits[:3]:
            t = clean_text(h.get("text",""))
            if t:
                texts.append(t)
        joined = "\n\n".join(texts)
        if joined and len(joined) > 120:
            try:
                out = summarizer(joined, max_length=80, min_length=20, do_sample=False)
                summ = out[0].get("summary_text","").strip()
                if summ:
                    if not re.search(r'[.!?]$', summ):
                        summ += '.'
                    return summ, None
            except Exception:
                pass

    return None, None

def synthesize_expanded(question, candidates):
    if not candidates or generator is None:
        return None, []
    facts = []
    sources = []
    used = set()
    for c in candidates[:6]:
        a = c.get("answer","").strip()
        key = re.sub(r'\s+', ' ', a.lower()).strip()
        if not key or key in used: continue
        used.add(key)
        facts.append("- " + a)
        s = c.get("source")
        if s and s not in sources: sources.append(s)
    if not facts:
        return None, []
    prompt = (
        "You are a helpful assistant. Using ONLY the facts below, write a friendly concise answer: "
        "1) one-line definition; 2) three short bullets 'what makes it special'; 3) three short bullets 'what you can build'; "
        "4) a tiny python example (if relevant).\n\n"
        f"Question: {question}\n\nFacts:\n" + "\n".join(facts) + "\n\nAnswer:"
    )
    try:
        out = generator(prompt, max_length=300, do_sample=False)
        text = out[0].get("generated_text","").strip()
        text = re.sub(r'\s{2,}', ' ', text)
        return text, sources[:2]
    except Exception:
        return None, sources[:2]

# ------------- Streamlit UI (minimal) -------------
st.set_page_config(page_title="Python Tutorial Chatbot", layout="centered")
st.title("🐍 Python Tutorial Chatbot")

download_from_gdrive_if_missing()
INDEX, CHUNKS = load_index_and_chunks()

if INDEX is None or CHUNKS is None:
    st.error("Index not available. Run embed_index.py locally and upload data/index.faiss & data/index.pkl or set GDRIVE vars.")
    st.stop()

q = st.text_input("Ask a question about the Python tutorial:", "")
if st.button("Ask") and q.strip():
    with st.spinner("Searching..."):
        hits = search_index(INDEX, CHUNKS, q, k=10)
        candidates = extractive_candidates(hits, q, keep_top=8)
        concise, source = get_concise_answer(q, hits, candidates)
        if concise:
            st.markdown("### Answer")
            st.write(concise)
            if source:
                st.markdown(f"**Source:** [{source}]({source})")
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
