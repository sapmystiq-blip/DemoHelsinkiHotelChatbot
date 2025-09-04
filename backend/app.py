import os
import json
import re
from difflib import SequenceMatcher
from typing import List, Tuple, Dict
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from difflib import SequenceMatcher
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()



""" Removing single kb.json file loading to support multiple files in a knowledgebase directory
HERE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(HERE, "kb.json")

try:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        KB = json.load(f)
except Exception:
    KB = [] 
    End of removal """

HERE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(HERE, "knowledgebase")

def load_kb():
    kb = []
    if os.path.exists(KB_DIR):
        for file in os.listdir(KB_DIR):
            if file.endswith(".json"):
                try:
                    with open(os.path.join(KB_DIR, file), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            kb.extend(data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    else:
        print(f"Knowledgebase folder not found at {KB_DIR}")  # debug
    return kb


KB = load_kb()

import math
from pydantic import BaseModel
from typing import List, Tuple

# --- Utilities ---
class ChatRequest(BaseModel):
    message: str

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9åäöáàâéèêíìîóòôúùû\- ]", " ", s)  # keep common letters/numbers
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize(s: str) -> List[str]:
    return [t for t in _normalize(s).split() if len(t) > 1]

# Simple lexical similarity: Jaccard + partial ratio
def _similarity(a: str, b: str) -> float:
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    jacc = len(ta & tb) / len(ta | tb)
    ratio = SequenceMatcher(a=_normalize(a), b=_normalize(b)).ratio()
    # Weighted blend; tune as needed
    return 0.55 * ratio + 0.45 * jacc

""" def find_best_kb_match(query: str, top_k: int = 5) -> List[Tuple[float, Dict[str, str]]]:
    scored = []
    for qa in KB:
        q = qa.get("question", "")
        if not q:
            continue
        scored.append((_similarity(query, q), qa))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k] """

def find_best_kb_match(query: str, top_k: int = 5) -> List[Tuple[float, Dict[str, str], str]]:
    scored = []
    q = query or ""
    for qa in KB:
        candidates = []
        if qa.get("question"):
            candidates.append(("question", qa["question"]))
        if qa.get("title"):
            candidates.append(("title", qa["title"]))
        if qa.get("answer"):
            candidates.append(("answer", qa["answer"]))  # NEW: also score the answer text

        for field, cand in candidates:
            s = _similarity(q, cand)
            scored.append((s, qa, cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# High-priority rules for common hotel intents (exact answers beat the LLM)
_RULES = [
    (re.compile(r"\b(check\s?-?out|checkout)\b.*\b(time|when)\b"), 
     "Check-out is until 11 AM."),
    (re.compile(r"\b(check\s?-?in|checkin)\b.*\b(time|when)\b"), 
     "Check-in starts at 3 PM."),
]

def rule_based_answer(user_msg: str) -> str | None:
    text = _normalize(user_msg)
    for pattern, answer in _RULES:
        if pattern.search(text):
            return answer
    return None

def retrieve_kb_context(user_msg: str, k: int = 6) -> list[dict]:
    """
    Return top-k UNIQUE KB Q&A dicts to ground the LLM.
    Normalizes outputs from find_best_kb_match() which may be tuples.
    """
    print(f"Retrieving KB context for: {user_msg}")  # debug

    # Over-fetch to improve diversity
    raw = find_best_kb_match(user_msg, top_k=max(k * 2, 6)) or []

    # Normalize to just the KB dicts (drop score/matched_text/etc.)
    items: list[dict] = []
    for row in raw:
        if isinstance(row, dict):
            items.append(row)
        elif isinstance(row, (list, tuple)):
            # (score, item, matched_text) OR (score, item)
            if len(row) >= 2 and isinstance(row[1], dict):
                items.append(row[1])
        # else: ignore strings/other types

    # Deduplicate by stable key (id > title > question)
    seen = set()
    unique: list[dict] = []
    for qa in items:
        key = qa.get("id") or qa.get("_id") or qa.get("title") or qa.get("question")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(qa)
        if len(unique) >= k:
            break

    return unique




app = FastAPI(title="Hotel Chatbot Demo", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

FRONTEND_DIR = os.path.abspath(os.path.join(HERE, "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

def normalize(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower())

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def find_best_answer(q: str) -> Dict[str, Any] | None:
    best, best_score = None, 0.0
    for item in KB:
        for field in [item.get("title", "")] + item.get("questions", []):
            s = sim(q, field)
            if s > best_score:
                best, best_score = item, s
    if best and best_score >= 0.55:
        return {"answer": best.get("answer",""), "title": best.get("title",""), "score": round(best_score,2)}
    return None

BOOKING_RE = re.compile(r"\b(book|reservation|reserve|room)\b", re.I)
CALLBACK_RE = re.compile(r"\b(call\s*back|phone\s*call|contact me|ring me)\b", re.I)

BOOKINGS: list[dict] = []
CALLBACKS: list[dict] = []

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message":"Frontend not found, but API is running."})



def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9åäöáàâéèêíìîóòôúùû\- ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize(s: str) -> List[str]:
    toks = [t for t in _normalize(s).split() if len(t) > 2]
    # naive singularization to reduce plural noise
    normed = []
    for t in toks:
        if t.endswith("s") and len(t) > 3:
            normed.append(t[:-1])
        else:
            normed.append(t)
    return normed


def _similarity(a: str, b: str) -> float:
    # Require some shared tokens to avoid weird matches
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / len(ta | tb)
    if overlap == 0:
        return 0.0
    char_ratio = SequenceMatcher(a=_normalize(a), b=_normalize(b)).ratio()
    return 0.55 * char_ratio + 0.45 * overlap

# Return (score, item, matched_text) so we can display what actually matched
def find_best_kb_match(query: str, top_k: int = 5) -> List[Tuple[float, Dict[str, str], str]]:
    scored: List[Tuple[float, Dict[str, str], str]] = []
    q = query or ""
    for qa in KB:
        candidates = []
        if qa.get("question"):
            candidates.append(qa["question"])
        if qa.get("title"):                 # keep title support for later use
            candidates.append(qa["title"])
        for cand in candidates:
            s = _similarity(q, cand)
            scored.append((s, qa, cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def _unique_by_id(items: List[Tuple[float, Dict[str, str], str]], k: int) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for _, qa, _ in items:
        key = id(qa)
        if key in seen:
            continue
        seen.add(key)
        out.append(qa)
        if len(out) >= k:
            break
    return out


@app.post("/api/chat")
def chat(req: ChatRequest):
    user_msg = (req.message or "").strip()

    # 1) Rules first (fast paths)
    rb = rule_based_answer(user_msg)
    if rb:
        print(f"Rule-based answer triggered.")  # debug
        return {"reply": rb}#, "source": "KB • rule"}
    
    # 2) Try KB direct match 
    top = find_best_kb_match(user_msg, top_k=5)
    # Slightly higher threshold reduces false positives; tune 0.65–0.72
    if top and top[0][0] >= 0.69:
        print(f"KB direct match score: {top[0][0]:.2f}")  # debug
        print(f"Matched text: {top[0][2]}")  # debug
        best_score, best_item, matched_text = top[0]
        return {
            "reply": best_item.get("answer", "I found a relevant entry but it has no answer.")#,
             #"source": f"KB • match {best_score:.2f} • {matched_text}"
        }
    

    # 3) Fall back to LLM, but ground it with the KB context
    """ context_items = retrieve_kb_context(user_msg, k=6)
    context_text = "\n\n".join(
        f"Q: {c.get('question') or c.get('title','')}\nA: {c.get('answer','')}"
        for c in context_items
    ) """
    context_items = retrieve_kb_context(user_msg, k=6)
    context_text = "\n\n".join(
    f"Q: {(c.get('question') or c.get('title') or '').strip()}\nA: {c.get('answer','').strip()}"
    for c in context_items
        if isinstance(c, dict)
)


    # If we have no context at all, keep the model cautious
    system_msg = (
        "You are a Helsinki hotel assistant. Prefer the provided Knowledge Base (KB) facts. "
        "Only answer using KB facts provided below. "
        "If the KB does not contain the answer, say: \"I don't have that specific information yet.\" "
        "Do not invent policies, prices, or times."
    )
    user_prompt = (
        f"User question:\n{user_msg}\n\n"
        f"KB context (use these facts only):\n{context_text if context_text else '[No KB context found]'}\n\n"
        "Answer concisely. If the KB does not cover it, say you don't have that specific information yet."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=350,
        )
        reply = response.choices[0].message.content
        src = "LLM • KB-grounded" if context_items else "LLM • no-KB"
        print(f"LLM reply: {reply}")  # debug
        print(f"Used context items: {len(context_items)}")  # debug
        print(f"Context:\n{context_text}")  # debug
        print(f"Source:\n{src}")  # debug
        return {"reply": reply}#, "source": src}
    except Exception as e:
        # Last-resort safety net
        return {"reply": f"LLM error: {e}"}#, "source": "system"}


@app.post("/api/booking")
def create_booking(payload: Dict[str, Any]):
    req = {k: payload.get(k) for k in ["name","email","check_in","check_out","guests","room_type"]}
    BOOKINGS.append(req)
    return {"status":"received","queue_length":len(BOOKINGS)}

@app.post("/api/callback")
def schedule_callback(payload: Dict[str, Any]):
    req = {k: payload.get(k) for k in ["name","phone","reason"]}
    CALLBACKS.append(req)
    return {"status":"received","queue_length":len(CALLBACKS)}
