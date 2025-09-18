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
GDRIVE_CHUNKS_ID = os.getenv("GDRIVE_CHUNKS_ID")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_FAISS = os.path.join(DATA_DIR, "index.faiss")
INDEX_PKL = os.path.join(DATA_DIR, "index.pkl")
CHUNKS_ID = os.path.join(DATA_DIR, "chunks.pkl")

def download_from_gdrive_if_missing():
    try:
        if GDRIVE_INDEX_FAISS_ID and not os.path.exists(INDEX_FAISS):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_INDEX_FAISS_ID}", INDEX_FAISS, quiet=True)
        if GDRIVE_INDEX_PKL_ID and not os.path.exists(INDEX_PKL):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_INDEX_PKL_ID}", INDEX_PKL, quiet=True)
        if GDRIVE_CHUNKS_ID and not os.path.exists(CHUNKS_PKL):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_CHUNKS_ID}", CHUNKS_PKL, quiet=True)
    except Exception:
        pass

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

def fix_encoding(s: str) -> str:
    if not s:
        return ""
    s = s.replace("Â", " ").replace("\xa0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

BOILERPLATE_PATTERNS = ["©", "Privacy", "Terms", "Search", "Menu", "Get Certified", "Sign In", "Try it Yourself"]
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

CANONICAL_MAP = {
    "oop": "object oriented programming",
    "object oriented programming": "object oriented programming",
    "object-oriented programming": "object oriented programming",
    "python": "python",
}

CURATED_FACTS = {
    "python": (
        "Python is a high-level, general-purpose programming language known for its simplicity and readability. "
        "It was created by Guido van Rossum and first released in 1991."
    ),
    "object oriented programming": (
        "Object-oriented programming (OOP) is a programming paradigm that uses 'objects'—data structures "
        "containing data and methods—to design programs and model real-world entities."
    ),
}

def canonicalize(q: str) -> str:
    if not q:
        return ""
    q0 = q.lower().strip()
    q0 = re.sub(r'[^a-z0-9\s\-]', ' ', q0)
    q0 = re.sub(r'\s+', ' ', q0).strip()
    return CANONICAL_MAP.get(q0, q0)

SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')
DEFINITION_VERBS = r"(?:\b(is|was|are|refers to|means|provides|represents|describes|allows|implements|supports)\b)"
BOOST_TERMS = [
    "programming language", "high-level", "object oriented", "object-oriented", "interpreted",
    "general-purpose", "designed to", "used to", "allows you to", "provides"
]

def score_sentence(sentence: str, canonical_keyword: str, base_score: float=0.0, dist: float=1e6) -> float:
    """Compute score for a candidate sentence. Higher = better."""
    s = sentence.lower()
    score = base_score
    if canonical_keyword and canonical_keyword in s:
        score += 2.5
    if re.search(rf"\b{re.escape(canonical_keyword)}\b.*{DEFINITION_VERBS}", s) or re.search(rf"{DEFINITION_VERBS}.*\b{re.escape(canonical_keyword)}\b", s):
        score += 4.0
    for t in BOOST_TERMS:
        if t in s:
            score += 1.5
    if len(sentence.strip()) < 30:
        score -= 2.0
    if sentence.strip().isupper():
        score -= 2.0
    try:
        score += (1.0 / (1.0 + dist)) * 2.0
    except Exception:
        pass
    return score

def find_best_definition(hits, user_query, top_k_hits=10):
    """
    Return (sentence, source) where sentence is the highest scoring candidate found in top hits.
    """
    canonical = canonicalize(user_query)
    candidates = []
    hits_sorted = sorted(hits, key=lambda x: x[1])[:top_k_hits]
    for h, dist in hits_sorted:
        text = clean_text(h.get("text",""))
        if not text:
            continue
        sents = SENT_SPLIT_RE.split(text)
        for s in sents:
            if not s or len(s.strip()) < 20:
                continue
            if len(s.strip().split()) <= 4 and s.strip().isupper():
                continue
            base = 0.0
            sc = score_sentence(s, canonical_keyword=canonical, base_score=base, dist=dist)
            if sc > -1.5: 
                candidates.append({"sent": s.strip(), "score": sc, "source": h.get("url","")})
    if not candidates:
        return None, None
    best = sorted(candidates, key=lambda x: -x["score"])[0]
    s = best["sent"]
    if not re.search(r'[.!?]$', s):
        s = s + '.'
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s, best.get("source")

def get_concise_answer(question, hits, candidates):
    sent, src = find_best_definition(hits, question, top_k_hits=10)
    if sent:
        return sent, src

    if candidates:
        top = candidates[0]
        one = top.get("answer","").strip()
        one = re.sub(r'(\b\w+\b)(?:\s+\1\b){2,}', r'\1', one)
        if not re.search(r'[.!?]$', one):
            one += '.'
        return one, top.get("source","")

    if summarizer and hits:
        texts = []
        for h, d in hits[:3]:
            t = clean_text(h.get("text",""))
            if t: texts.append(t)
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

    canonical = canonicalize(question)
    if canonical in CURATED_FACTS:
        return CURATED_FACTS[canonical], "general knowledge"

    if generator and candidates:
        facts = []
        used = set()
        for c in candidates[:6]:
            a = c.get("answer","").strip()
            key = re.sub(r'\s+', ' ', a.lower()).strip()
            if not key or key in used: continue
            used.add(key)
            facts.append("- " + a)
        if facts:
            prompt = (
                "Using ONLY the facts below, write one short definition (one sentence) for the requested term. "
                "If facts are insufficient, say 'I couldn't find this in the site data.'\n\n"
                "Facts:\n" + "\n".join(facts) + "\n\nDefinition:"
            )
            try:
                out = generator(prompt, max_length=80, do_sample=False)
                gen = out[0].get("generated_text","").strip()
                if gen:
                    first_line = gen.splitlines()[0].strip()
                    if not re.search(r'[.!?]$', first_line):
                        first_line += '.'
                    return first_line, None
            except Exception:
                pass

    return None, None

def synthesize_expanded(question, candidates):
    """Generate a fuller answer using the generator (friendly multi-part)."""
    if not candidates or generator is None:
        return None, []
    facts = []
    sources = []
    used = set()
    for c in candidates[:8]:
        a = c.get("answer","").strip()
        k = re.sub(r'\s+', ' ', a.lower()).strip()
        if not k or k in used: continue
        used.add(k)
        facts.append("- " + a)
        s = c.get("source")
        if s and s not in sources:
            sources.append(s)
    if not facts:
        return None, []
    prompt = (
        "You are a helpful assistant. Using ONLY the facts below, write a friendly concise answer structured as:\n"
        "1) One-line definition; 2) three short bullets 'what makes it special'; 3) three short bullets 'what you can build'; "
        "4) a tiny python example (if relevant).\n\n"
        f"Question: {question}\n\nFacts:\n" + "\n".join(facts) + "\n\nAnswer:"
    )
    try:
        out = generator(prompt, max_length=320, do_sample=False)
        text = out[0].get("generated_text","").strip()
        text = re.sub(r'\s{2,}', ' ', text)
        return text, sources[:2]
    except Exception:
        return None, sources[:2]

st.set_page_config(page_title="Python Tutorial Chatbot", layout="centered")
st.title("🐍 Python Tutorial Chatbot")

download_from_gdrive_if_missing()
INDEX, CHUNKS = load_index_and_chunks()

if INDEX is None or CHUNKS is None:
    st.error("Index files missing. Run embed_index.py locally, upload data/index.faiss and data/index.pkl, or set GDRIVE_INDEX_FAISS_ID & GDRIVE_INDEX_PKL_ID.")
    st.stop()

query = st.text_input("Ask a question about the Python tutorial:", "")
if st.button("Ask") and query.strip():
    with st.spinner("Searching..."):
        hits = search_index(INDEX, CHUNKS, query, k=10)
        candidates = extractive_candidates(hits, query, keep_top=8)
        concise, source = get_concise_answer(query, hits, candidates)
        if concise:
            st.markdown("### Answer")
            st.write(concise)
            if source:
                st.markdown(f"**Source:** {source}" if source == "general knowledge" else f"**Source:** [{source}]({source})")
            if st.button("Show expanded answer"):
                expanded, sources = synthesize_expanded(query, candidates)
                if expanded:
                    st.markdown("### Expanded Answer")
                    st.write(expanded)
                    if sources:
                        st.markdown("**Sources:** " + ", ".join(f"[{s}]({s})" for s in sources))
                else:
                    st.info("No expanded answer available.")
        else:
            st.info("I couldn't find a confident answer in the indexed content.")
