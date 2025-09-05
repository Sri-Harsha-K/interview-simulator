import os, json
from typing import List, Dict, Any, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Simple alias for chat messages like {"role": "system"|"user"|"assistant", "content": "..."}
Message = Dict[str, Any]

class HttpError(Exception):
    ...

def _qwen_chatml(messages: List[Message]) -> str:
    """
    Build a Qwen-style ChatML prompt:
    <|im_start|>system ... <|im_end|><|im_start|>user ... <|im_end|>...<|im_start|>assistant
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role not in ("system", "user", "assistant"):
            role = "user"
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")  # generation starts here
    return "".join(parts)

class HFInferenceClient:
    """
    Calls Hugging Face Inference API:
      POST https://api-inference.huggingface.co/models/{model}
    Payload:
      {"inputs": "<chatml prompt>", "parameters": {"max_new_tokens": N, "temperature": T, "return_full_text": false}}
    Returns:
      str (assistant text)
    """
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        self.default_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
        self.timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
        if not self.api_key or not self.model:
            raise ValueError("LLM_API_KEY and LLM_MODEL must be set in .env")
        self.url = f"https://api-inference.huggingface.co/models/{self.model}"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(HttpError),
    )
    def _post(self, payload: dict) -> httpx.Response:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(self.url, headers=self.headers, json=payload)
        except httpx.HTTPError as e:
            raise HttpError(f"Network error: {e}") from e
        if r.status_code >= 500:
            # trigger retry on transient server errors
            raise HttpError(f"Upstream {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            # don't retry on permanent 4xx
            raise RuntimeError(f"HF Inference API {r.status_code}: {r.text[:300]}")
        return r

    def complete_chat(self, messages: List[Message], max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        prompt = _qwen_chatml(messages)
        params = {
            "max_new_tokens": max_tokens or self.default_max_tokens,
            "temperature": float(temperature),
            "return_full_text": False,
        }
        payload = {"inputs": prompt, "parameters": params}
        data = self._post(payload).json()
        # Common Inference API shapes:
        #   [{"generated_text": "..."}]  OR  {"generated_text": "..."}
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        raise RuntimeError(f"Unexpected HF response: {json.dumps(data)[:300]}")
