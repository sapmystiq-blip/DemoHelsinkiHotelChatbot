# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os

# Optional: load .env for OPENAI_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- FastAPI app ---
app = FastAPI(title="Hotel Chatbot API", version="0.1.0")

# CORS (loose for dev; tighten allow_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["https://your-site.com"] in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static site (your demo UI) ---
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def root():
    return FileResponse("app/static/index.html")

# --- Models ---
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = None  # optional: [{"role":"user|assistant|system","content":"..."}]

class ChatResponse(BaseModel):
    reply: str

# --- Healthcheck ---
@app.get("/health")
def health():
    return {"status": "ok"}

# --- Chat endpoint ---
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    If OPENAI_API_KEY is set in env/.env -> calls OpenAI.
    Otherwise, returns a simple echo so the UI can be tested without keys.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    # No key? Echo so the frontend works during setup.
    if not api_key:
        return ChatResponse(reply=f"(demo echo) You said: {req.message}")

    # OpenAI client
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        # If the SDK isn't installed, hint clearly
        raise HTTPException(status_code=500, detail=f"OpenAI SDK not available: {e}")

    # Build messages (lightweight, you can expand later)
    messages = [
        {"role": "system", "content": "You are a concise, helpful hotel assistant."},
    ]
    if req.history:
        messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        reply = resp.choices[0].message.content
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
