# backend/app.py
from __future__ import annotations

import os
import re
import json
import math
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import Counter

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from difflib import SequenceMatcher

# Optional LLM (OpenAI) client
OPENAI_CLIENT = None
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_ENABLED = False
LLM_TIMEOUT_SECS = int(os.getenv("LLM_TIMEOUT_SECS", "20"))
PRIMARY_LANG = os.getenv("PRIMARY_LANG", "fi")
LANGUAGE_POLICY = os.getenv("LANGUAGE_POLICY", "always_primary")  # always_primary | match_user
SUPPORTED_LANG_HINT = "fi, en, sv, no, de, fr, es, it"

# Optional: load .env (OPENAI_API_KEY, HOST, PORT, etc.)
try:
    from dotenv import load_dotenv  # python-dotenv
    load_dotenv()
except Exception:
    pass

# Initialize OpenAI client if key present (optional)
try:
    from openai import OpenAI  # type: ignore
    if os.getenv("OPENAI_API_KEY"):
        OPENAI_CLIENT = OpenAI()
        LLM_ENABLED = os.getenv("LLM_ENABLED", "true").lower() in {"1","true","yes","on"}
    else:
        LLM_ENABLED = False
except Exception:
    OPENAI_CLIENT = None
    LLM_ENABLED = False

# ============================================================
# Language detection (optional)
# ============================================================
try:
    from langdetect import detect as _ld_detect  # type: ignore
    def detect_lang(text: str) -> str:
        try:
            return _ld_detect(text) or "en"
        except Exception:
            return "en"
except Exception:
    def detect_lang(text: str) -> str:
        return "en"

LANG_NAMES = {
    "en": "English", "fi": "Finnish", "sv": "Swedish", "no": "Norwegian",
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "da": "Danish"
}

logger = logging.getLogger("uvicorn")

# ============================================================
# Paths
# ============================================================
HERE = Path(__file__).resolve().parent           # repo/backend/
REPO_ROOT = HERE.parent                          # repo/
FRONTEND_DIR = REPO_ROOT / "frontend"            # index.html, styles.css, chat.js
INDEX_FILE = FRONTEND_DIR / "index.html"

# KB JSON files live under backend/knowledgebase
KB_DIR = HERE / "knowledgebase"
# Discover all JSON KB files using glob (no hardcoding)
KB_FILES = [p.name for p in sorted(KB_DIR.glob("*.json"))]

# ============================================================
# Database (optional; e.g., Railway Postgres)
# ============================================================
DB_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
DB_ENABLED = bool(DB_URL)
ENGINE = None
TABLE_READY = False

def _db_connect_and_prepare():
    """Initialize DB connection and ensure schema exists.
    Uses SQLAlchemy Core to be lightweight.
    """
    global ENGINE, TABLE_READY
    if not DB_ENABLED or ENGINE is not None:
        return
    try:
        from sqlalchemy import create_engine, text
        # Default pool size is fine for single worker; Railway uses one dyno
        ENGINE = create_engine(DB_URL, pool_pre_ping=True)
        with ENGINE.begin() as conn:
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                  id BIGSERIAL PRIMARY KEY,
                  session_id TEXT,
                  role TEXT NOT NULL,
                  message TEXT NOT NULL,
                  source TEXT,
                  match_score DOUBLE PRECISION,
                  created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            ))
        TABLE_READY = True
        logger.info("DB initialized: chat_messages table ready")
    except Exception as e:
        logger.exception(f"DB init failed: {e}")
        ENGINE = None
        TABLE_READY = False

def _db_insert_message(session_id: str | None, role: str, message: str, source: str | None, match_score: float | None):
    if not ENGINE or not TABLE_READY:
        return
    try:
        from sqlalchemy import text
        with ENGINE.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO chat_messages (session_id, role, message, source, match_score)
                    VALUES (:sid, :role, :msg, :src, :ms)
                """),
                {"sid": session_id, "role": role, "msg": message, "src": source, "ms": match_score}
            )
    except Exception as e:
        logger.warning(f"DB insert failed: {e}")

# ============================================================
# Retrieval acceptance gates
# ============================================================
MIN_ACCEPT_SCORE = 1.20   # overall blend threshold (tune 1.0–2.0)
MIN_BM25_SIGNAL  = 0.10   # require some keyword signal
MIN_JACCARD      = 0.05   # or some token overlap
MIN_FUZZY        = 0.40   # or moderate fuzzy similarity

# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="Hotel Chatbot Demo")

# ============================================================
# Models
# ============================================================
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    reply: str
    source: str | None = None
    match: float | None = None
    session_id: str | None = None

# ============================================================
# Text normalization / tokenization
# ============================================================
STOPWORDS = {
    # EN
    "the","a","an","and","or","to","of","for","on","in","is","it","are","do","you","we","i",
    "can","with","at","my","our","your","me","us","be","have","has","will","from","about",
    # common question words
    "what","which","where","when","who","whom","whose","how",
    # FI (tiny set; expand as needed)
    "ja","tai","se","ne","että","kuin","minun","meidän","teidän","sinun","oma","olen","ovat",
}

SYNONYMS = {
    # English variations
    "wifi":"wi-fi", "wi-fi":"wifi", "internet":"wifi",
    "parking":"car park", "carpark":"car park",
    "pool":"swimming pool", "swimming":"pool",
    "gym":"fitness", "fitness":"fitness",
    "checkout":"check-out", "checkin":"check-in",
    "late checkout":"late check-out", "late-checkout":"late check-out",

    # Sightseeing / attractions phrasing
    "sightsee":"attractions", "sightseeing":"attractions", "sights":"attractions",
    "attraction":"attractions", "landmarks":"attractions", "places":"attractions",
    "visit":"attractions", "doing":"attractions", "do":"attractions",

    # Finnish → English mapping (compact)
    "aamiainen":"breakfast",
    "pysäköinti":"parking",
    "sauna":"sauna",
    "uima-allas":"pool",
    "kuntosali":"gym",
    "myöhäinen":"late",
    "myöhäinen uloskirjautuminen":"late check-out",
    "sisäänkirjautuminen":"check-in",
    "uloskirjautuminen":"check-out",
}

def _normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s\-]", " ", t)  # drop punctuation
    t = re.sub(r"\s+", " ", t)
    return t

def _tokens(text: str):
    for tok in _normalize(text).split():
        if tok in STOPWORDS:
            continue
        tok = SYNONYMS.get(tok, tok)  # map synonyms
        yield tok

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()

def _token_overlap(a: str, b: str) -> float:
    sa = set(_tokens(a)); sb = set(_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

# ============================================================
# KB + Index (mutable; rebuilt on startup)
# ============================================================
KB: List[Dict[str, Any]] = []
DOCS: List[Dict[str, Any]] = []
DF: Counter = Counter()
N: int = 0
AVG_LEN: float = 0.0

def load_kb_clean() -> List[Dict[str, Any]]:
    kb: List[Dict[str, Any]] = []
    seen = set()
    logger.info(f"Looking for KB files in: {KB_DIR.resolve()}")
    for fname in KB_FILES:
        fpath = KB_DIR / fname
        if not fpath.exists():
            logger.warning(f"KB file not found, skipping: {fname}")
            continue
        try:
            logger.info(f"Loading KB file: {fname}")
            data = json.loads(fpath.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                logger.warning(f"KB file is not a list, skipping: {fname}")
                continue
            for row in data:
                if not isinstance(row, dict):
                    continue
                q = (row.get("question") or "").strip()
                a = (row.get("answer") or "").strip()
                if not q or not a:
                    continue
                key = (_normalize(q), _normalize(a))
                if key in seen:
                    continue
                seen.add(key)
                kb.append({
                    "question": q,
                    "answer": a,
                    "title": row.get("title") or "",
                    "file": fname
                })
        except Exception as e:
            logger.exception(f"Error reading KB file {fname}: {e}")
            continue
    logger.info(f"Loaded {len(kb)} KB entries from {len(KB_FILES)} files")
    return kb

def build_index(kb: List[Dict[str, Any]]):
    global DOCS, DF, N, AVG_LEN
    DOCS = []
    DF = Counter()
    for i, item in enumerate(kb):
        text = f"{item.get('question','')} {item.get('answer','')}"
        toks = list(_tokens(text))
        DOCS.append({"id": i, "toks": toks, "len": len(toks)})
        for t in set(toks):
            DF[t] += 1
    N = len(kb)
    AVG_LEN = (sum(d["len"] for d in DOCS) / max(1, len(DOCS))) if DOCS else 0.0
    logger.info(f"Indexed {N} KB docs. AVG_LEN={AVG_LEN:.2f}, vocab={len(DF)}")

def _bm25_score(query_tokens: List[str], doc) -> float:
    # BM25 parameters
    k1, b = 1.4, 0.75
    score = 0.0
    if not doc["toks"]:
        return 0.0
    tf = Counter(doc["toks"])
    for qt in query_tokens:
        df = DF.get(qt, 0)
        if df == 0:
            continue
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        f = tf.get(qt, 0)
        denom = f + k1 * (1 - b + b * (doc["len"] / (AVG_LEN or 1)))
        score += idf * ((f * (k1 + 1)) / (denom or 1))
    return score

def expand_query(q: str) -> List[str]:
    base = list(_tokens(q))
    expanded = []
    for t in base:
        expanded.append(t)
        if t in SYNONYMS:
            expanded.append(SYNONYMS[t])  # one-hop
    return list(dict.fromkeys(expanded))  # de-dupe, order-preserving

def score_item(query: str, item: Dict[str, Any], doc):
    qtokens = expand_query(query)
    bm25 = _bm25_score(qtokens, doc)
    fuzzy = _ratio(query, item["question"])
    jacc  = _token_overlap(query, item["question"])
    blend = (0.62 * bm25) + (0.28 * fuzzy) + (0.10 * jacc)
    return blend, bm25, fuzzy, jacc

def find_best_kb_match(query: str, top_k: int = 3):
    if not KB:
        return []

    nq = _normalize(query)

    # Exact question match → return very strong score so it passes gates
    for it in KB:
        if _normalize(it["question"]) == nq:
            return [(10.0, 10.0, 1.0, 1.0, it)]  # blend, bm25, fuzzy, jacc, item

    scored = []
    for d in DOCS:
        it = KB[d["id"]]
        blend, bm25, fuzzy, jacc = score_item(query, it, d)
        if blend > 0:
            scored.append((blend, bm25, fuzzy, jacc, it))

    scored.sort(key=lambda x: x[0], reverse=True)

    # de-dup by identical answers to avoid spam ties
    seen_ans, deduped = set(), []
    for blend, bm25, fuzzy, jacc, it in scored:
        k = _normalize(it["answer"])
        if k in seen_ans:
            continue
        seen_ans.add(k)
        deduped.append((blend, bm25, fuzzy, jacc, it))
        if len(deduped) >= top_k:
            break
    return deduped

# ============================================================
# Rule-based fast paths (greetings, intents)
# ============================================================
BOOKING_KEYWORDS = {"book","reservation","reserve","varaa","booking"}
CALLBACK_KEYWORDS = {"callback","call back","phone call","ring me","soita"}
HELP_KEYWORDS = {"help","apua","support","human","agent"}
GREETINGS = {"hei","moi","terve","hi","hello","hey","hola","ciao"}

def rule_based_answer(user_msg: str) -> str | None:
    text = _normalize(user_msg)

    # greetings (exact or startswith greeting)
    if text in GREETINGS or any(text.startswith(g + " ") for g in GREETINGS):
        return "Hei! 👋 Kuinka voin auttaa? (Hi! How can I help?)"

    # thanks
    if any(p in text for p in {"thanks","thank you","kiitos"}):
        return "Ole hyvä! (You’re welcome.)"

    # booking intent (Finnish-first)
    if any(k in text for k in BOOKING_KEYWORDS):
        return (
            "Voin auttaa varauksen aloittamisessa. Kerrothan: \n"
            "• Saapumispäivä\n• Lähtöpäivä\n• Vieraat (aikuiset/lapset)\n"
            "• Huonetyyppitoive (esim. twin, double, perhe)"
        )

    # callback intent (Finnish-first)
    if any(k in text for k in CALLBACK_KEYWORDS):
        return (
            "Voin järjestää takaisinsoiton. Anna: \n"
            "• Nimesi\n• Puhelinnumero (maatunnus)\n"
            "• Toivottu aikaikkuna (aikavyöhyke)\n• Aihe (varaus, lasku, ryhmät)"
        )

    # help / human handoff (Finnish-first)
    if any(k in text for k in HELP_KEYWORDS):
        return "Autan mielelläni. Halutessasi voin ohjata tämän ihmiselle."

    return None

# ============================================================
# Fallback composer (out-of-scope aware)
# ============================================================
def llm_like_answer(query: str, kb_items: List[Dict[str, Any]], respond_lang: str | None = None) -> str:
    toks = list(_tokens(query))
    has_overlap = any(t in DF for t in toks)
    if not kb_items or not has_overlap:
        if (respond_lang or PRIMARY_LANG) == "fi":
            return (
                f"En löytänyt tietoja aiheesta “{query}”. "
                "Voin auttaa hotelliin liittyvissä asioissa, kuten sisään‑/uloskirjautuminen, aamiainen, pysäköinti, huoneet, sauna, kuntosali ja maksut. "
                "Kokeile kysyä: “Mihin aikaan on aamiainen?”, “Onko pysäköintiä?”, tai “Voinko maksaa käteisellä?”."
            )
        else:
            return (
                f"I couldn’t find details about “{query}”. "
                "I can help with hotel topics like check-in/out, breakfast, parking, rooms, sauna, gym, and payments. "
                "Try asking: “What time is breakfast?”, “Do you have parking?”, or “Can I pay in cash?”."
            )
    best = kb_items[0]
    ans = (best.get("answer") or "").strip()
    return ans or "I found a related item, but it had no answer text."

def generate_llm_answer(query: str, kb_items: List[Dict[str, Any]], respond_lang: str) -> str:
    """Use OpenAI to compose a KB-grounded answer in the requested language.
    Safe fallback if unavailable.
    """
    if not OPENAI_CLIENT or not LLM_ENABLED:
        return llm_like_answer(query, kb_items, respond_lang)

    # Light intent filtering to avoid mixing unrelated topics (e.g., park vs parking)
    def _infer_intent(text: str) -> str | None:
        t = _normalize(text)
        toks = set(_tokens(text))
        # Parking (vehicle) intent heuristics
        if (
            "parking" in toks
            or "car" in toks and "park" in t.split()
            or "garage" in t
            or "pysäköinti" in t
            or "car park" in t
        ):
            return "parking"
        return None

    def _filter_items_for_intent(intent: str | None, items: List[Dict[str, Any]]):
        if not intent:
            return items
        if intent == "parking":
            keep_kw = {"parking", "garage", "car park", "pysäköinti", "charging", "ev", "electric"}
            filtered: List[Dict[str, Any]] = []
            for it in items:
                text = _normalize(f"{it.get('question','')} {it.get('answer','')}")
                if any(kw in text for kw in keep_kw):
                    filtered.append(it)
            # If filtering removed everything, fall back to originals
            return filtered or items
        return items

    intent = _infer_intent(query)
    kb_items = _filter_items_for_intent(intent, kb_items)

    # Build compact context from top items
    context_blocks = []
    for it in kb_items[:5]:
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        if not a:
            continue
        src = it.get("file") or "knowledgebase"
        context_blocks.append(f"Q: {q}\nA: {a}\n(Source: {src})")

    if not context_blocks:
        return llm_like_answer(query, kb_items, respond_lang)

    lang_name = LANG_NAMES.get(respond_lang, respond_lang)
    system = (
        "You are a helpful Helsinki hotel assistant. Answer strictly using the provided knowledge base. "
        "Only answer the user's explicit intent and do not add unrelated sightseeing or park recommendations unless asked. "
        "If the information is missing, say you don't know and recommend contacting the front desk. "
        f"Be concise, friendly, and do not invent details. Respond in {lang_name}."
    )
    context = "\n\n".join(context_blocks)
    user = (
        f"User question: {query}\n\n"
        f"Knowledge base excerpts:\n{context}\n\n"
        "Compose the best possible answer using only the excerpts above."
    )

    try:
        resp = OPENAI_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=300,
            timeout=LLM_TIMEOUT_SECS,
        )
        txt = (resp.choices[0].message.content or "").strip()
        return txt or llm_like_answer(query, kb_items, respond_lang)
    except Exception:
        return llm_like_answer(query, kb_items, respond_lang)

# ============================================================
# API routes
# ============================================================
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "kb_items": len(KB),
        "kb_dir": str(KB_DIR.resolve()),
        "vocab": len(DF),
        "avg_len": round(AVG_LEN, 2),
        "llm_enabled": bool(LLM_ENABLED),
        "llm_model": LLM_MODEL if LLM_ENABLED else None,
        "langdetect": True,
        "lang_hint": SUPPORTED_LANG_HINT,
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, response: Response):
    user_msg = (req.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Session handling: prefer payload session_id; else cookie; else create
    session_id = (req.session_id or request.cookies.get("chat_session") or "").strip()
    if not session_id:
        import uuid
        session_id = uuid.uuid4().hex
        # 30 days cookie
        response.set_cookie("chat_session", session_id, max_age=60*60*24*30, httponly=False, samesite="Lax")

    # Language handling
    user_lang = detect_lang(user_msg)
    respond_lang = PRIMARY_LANG if LANGUAGE_POLICY == "always_primary" else user_lang

    # Log user message
    try:
        _db_insert_message(session_id, "user", user_msg, None, None)
    except Exception:
        pass

    # 1) Rules first
    rb = rule_based_answer(user_msg)
    if rb:
        return ChatResponse(reply=rb, source="Rules", match=1.0)

    # 2) Retrieve
    top = find_best_kb_match(user_msg, top_k=3)

    if top:
        best_blend, bm25, fuzzy, jacc, best_item = top[0]

        # Require some lexical evidence (BM25 or Jaccard). Fuzzy alone is not enough.
        strong_enough = (
            best_blend >= MIN_ACCEPT_SCORE or
            (bm25 >= MIN_BM25_SIGNAL) or
            (jacc >= MIN_JACCARD) or
            (fuzzy >= max(MIN_FUZZY, 0.55) and (bm25 > 0 or jacc > 0))
        )

        if strong_enough:
            kb_items = [t[4] for t in top]
            if LLM_ENABLED and OPENAI_CLIENT:
                reply = generate_llm_answer(user_msg, kb_items, respond_lang=respond_lang)
                src = f"LLM • KB-grounded ({best_item.get('file','')})"
            else:
                reply = (best_item.get("answer") or "").strip() or llm_like_answer(user_msg, kb_items, respond_lang)
                src = f"KB • {best_item.get('file','')}"
            # Log assistant message
            try:
                _db_insert_message(session_id, "assistant", reply, src, float(round(best_blend, 3)))
            except Exception:
                pass
            return ChatResponse(reply=reply, source=src, match=float(round(best_blend, 3)), session_id=session_id)

    # 3) Out-of-scope / soft fallback
    kb_items = [t[4] for t in top] if top else []
    if LLM_ENABLED and OPENAI_CLIENT:
        reply = generate_llm_answer(user_msg, kb_items, respond_lang=respond_lang)
        src = "LLM • Fallback KB-grounded"
    else:
        reply = llm_like_answer(user_msg, kb_items, respond_lang)
        src = "Fallback • KB-grounded"
    best_score = float(round(top[0][0], 3)) if top else 0.0
    try:
        _db_insert_message(session_id, "assistant", reply, src, best_score)
    except Exception:
        pass
    return ChatResponse(reply=reply, source=src, match=best_score, session_id=session_id)

# ============================================================
# Startup: load KB and build index (with logs)
# ============================================================
@app.on_event("startup")
def startup_event():
    global KB
    logger.info("=== App startup: loading KB and building index ===")
    if DB_ENABLED:
        _db_connect_and_prepare()
    KB = load_kb_clean()
    build_index(KB)

# ============================================================
# Frontend routes (serve index.html + static assets from /frontend)
# ============================================================
if not FRONTEND_DIR.exists():
    raise RuntimeError(f"Frontend folder not found at: {FRONTEND_DIR}")
if not INDEX_FILE.exists():
    raise RuntimeError(f"index.html not found at: {INDEX_FILE}")

@app.get("/")
def root():
    return FileResponse(INDEX_FILE)

# Mount ALL static assets at site root (keep AFTER API routes)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# ============================================================
# Local dev entrypoint
# ============================================================
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    import uvicorn
    uvicorn.run("backend.app:app", host=host, port=port, reload=True)
