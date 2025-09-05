from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

Message = Dict[str, Any]  # {"role": "system"|"user"|"assistant", "content": str}

@dataclass
class PersonaConfig:
    name: str
    system_prompt: str

class BaseAgent:
    """Shared behavior for personas: build messages, ask question, give feedback."""
    def __init__(self, cfg: PersonaConfig, llm) -> None:
        self.cfg = cfg
        self.llm = llm

    def _messages(self, user_content: str) -> List[Message]:
        return [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user",   "content": user_content},
        ]

    def ask(self, context: str) -> str:
        """Produce ONE interview question based on the context."""
        prompt = (
            "Given the candidate/context, ask exactly ONE interview question.\n"
            "Be concise (≤30 words). No preface, no numbering.\n\n"
            f"Context:\n{context}"
        )
        return self.llm.complete_chat(self._messages(prompt), max_tokens=128, temperature=0.7).strip()

    def feedback(self, question: str, answer: str) -> str:
        """Return 4 short bullets of feedback (or a prompt to answer if blank)."""
        # If the candidate submits nothing (or only whitespace), skip the LLM call.
        if not answer or not answer.strip():
            return "Please Answer the question"
    
        prompt = (
            "Identify and express the positive points and Evaluate the candidate’s answer in 4 short bullets covering:\n"
            "1) positives, 2) structure, 3) technical depth/accuracy, 4) trade-offs.\n"
            "Each bullet ≤20 words. No preface text.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}"
        )
        return self.llm.complete_chat(
            self._messages(prompt), max_tokens=160, temperature=0.6
        ).strip()
