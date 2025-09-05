# backend/models/http_client.py
import os
import json
from typing import List, Dict, Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

Message = Dict[str, Any]  # {"role": "system"|"user"|"assistant", "content": str}


class HttpError(Exception):
    ...


class HttpLLMClient:
    """
    OpenAI-compatible client for /v1/chat/completions.
    Works with:
      - Hugging Face Router: https://router.huggingface.co/v1
      - OpenAI:              https://api.openai.com/v1
      - Any other provider exposing the same API
    """

    def __init__(self) -> None:
        # Required
        self.base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "")

        # Optional
        self.default_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "128"))
        self.timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        self.stream_default = os.getenv("LLM_STREAM", "false").lower() == "true"
        self.org = os.getenv("OPENAI_ORG", "") or os.getenv("OPENAI_ORGANIZATION", "")

        # Optional fallback (used for e.g., insufficient_quota on primary)
        self.fb_base_url = (os.getenv("LLM_FALLBACK_BASE_URL", "") or "").rstrip("/")
        self.fb_api_key = os.getenv("LLM_FALLBACK_API_KEY", "")
        self.fb_model = os.getenv("LLM_FALLBACK_MODEL", "")

        if not self.base_url or not self.api_key or not self.model:
            raise ValueError("LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL must be set in .env")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.org:
            self.headers["OpenAI-Organization"] = self.org

    # ---------- internal HTTP helpers ----------

    def _endpoint(self, base_url: Optional[str] = None) -> str:
        base = (base_url or self.base_url).rstrip("/")
        return f"{base}/chat/completions"

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(HttpError),
    )
    def _post(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> httpx.Response:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as e:
            # Network / DNS / TLS errors → retry
            raise HttpError(f"Network error: {e}") from e

        # 5xx errors → retry
        if 500 <= r.status_code < 600:
            raise HttpError(f"Upstream {r.status_code}: {r.text[:300]}")
        return r

    def _try_fallback_if_quota(self, r: httpx.Response, messages: List[Message], max_tokens: int, temperature: float) -> httpx.Response:
        """If primary returns insufficient_quota (429), try fallback (if configured)."""
        if r.status_code != 429:
            return r
        if not (self.fb_base_url and self.fb_api_key and self.fb_model):
            return r

        try:
            err = r.json().get("error", {})
        except Exception:
            return r

        if err.get("code") == "insufficient_quota":
            fb_headers = {
                "Authorization": f"Bearer {self.fb_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.fb_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": float(temperature),
                "stream": False,
            }
            return self._post(self._endpoint(self.fb_base_url), fb_headers, payload)

        return r

    # ---------- public API ----------

    def complete_chat(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Send chat messages to an OpenAI-compatible /chat/completions endpoint and return the text.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of {role, content} dicts")

        use_stream = self.stream_default if stream is None else bool(stream)
        if use_stream:
            # You can add streaming support later; for now enforce non-streaming
            use_stream = False

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens or self.default_max_tokens),
            "temperature": float(temperature),
            "stream": use_stream,
        }

        r = self._post(self._endpoint(), self.headers, payload)
        r = self._try_fallback_if_quota(r, messages, payload["max_tokens"], payload["temperature"])

        if r.status_code >= 400:
            # Surface a concise, helpful error
            try:
                data = r.json()
                raise RuntimeError(f"LLM API {r.status_code}: {json.dumps(data)[:300]}")
            except Exception:
                raise RuntimeError(f"LLM API {r.status_code}: {r.text[:300]}")

        data = r.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected response shape: {json.dumps(data)[:300]}") from e

    # Optional: keep compatibility with older 'complete(prompt)' signature
    def complete(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.complete_chat(messages, **kwargs)
