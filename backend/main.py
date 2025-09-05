# backend/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.orchestrator import Orchestrator

# Load .env first thing
load_dotenv()

# Choose LLM client by env
PROVIDER = os.getenv("LLM_PROVIDER", "http")  # "http" (OpenAI-compatible) or "hf_inference"

if PROVIDER == "http":
    # Uses OpenAI-compatible /v1/chat/completions (e.g., HF Router, OpenAI, etc.)
    from backend.models.http_client import HttpLLMClient as Client
elif PROVIDER == "hf_inference":
    # Uses Hugging Face Inference API /models/<model> with ChatML prompt building
    from backend.models.hf_inference_client import HFInferenceClient as Client
else:
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {PROVIDER}")

app = FastAPI(title="Interview Simulator API", version="0.1.0")

# Instantiate LLM + orchestrator
try:
    llm = Client()
except Exception as e:
    # Fail early with a clear message (missing env vars, etc.)
    raise RuntimeError(f"Failed to initialize LLM client: {e}") from e

orch = Orchestrator(llm)

# ----- Schemas -----
class AskReq(BaseModel):
    context: str = ""

class EvalReq(BaseModel):
    persona: str
    question: str
    answer: str

# ----- Endpoints -----
@app.get("/")
def root():
    """Quick status + what the server actually loaded."""
    return {
        "ok": True,
        "service": "interview-sim",
        "provider": PROVIDER,
        "model": os.getenv("LLM_MODEL"),
        "client": type(llm).__name__,
    }

@app.post("/ask")
def ask(req: AskReq):
    try:
        return orch.ask(req.context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/ask failed: {e}")

@app.post("/evaluate")
def evaluate(req: EvalReq):
    try:
        return orch.evaluate(req.persona, req.question, req.answer)
    except KeyError:
        raise HTTPException(status_code=400, detail="Unknown persona key.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/evaluate failed: {e}")

# Optional: run directly (you can still launch with `uvicorn backend.main:app --reload --port 8000`)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=int(os.getenv("APP_PORT", "8000")), reload=True)
