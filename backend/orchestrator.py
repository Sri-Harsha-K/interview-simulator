# backend/orchestrator.py
"""
Turns the LLM into a small panel of interview personas (tech, HR, mentor).
- Rotates personas on each /ask call (tech -> hr -> mentor -> tech -> ...)
- Uses the same persona to generate feedback for a given question/answer
"""

from __future__ import annotations
from typing import Dict

# Relative imports (because this module is imported as `backend.orchestrator`)
from .agents.tech import TechAgent, TECH_CFG
from .agents.hr import HRAgent, HR_CFG
from .agents.mentor import MentorAgent, MENTOR_CFG


class Orchestrator:
    """Coordinates which persona asks next and returns structured results."""

    def __init__(self, llm) -> None:
        # Instantiate all personas with the same LLM client
        self.personas: Dict[str, object] = {
            "tech":   TechAgent(TECH_CFG, llm),
            "hr":     HRAgent(HR_CFG, llm),
            "mentor": MentorAgent(MENTOR_CFG, llm),
        }
        # Default rotation order
        self._order = ["tech", "hr", "mentor"]
        self._turn = 0

    # ----- rotation helpers -----
    def _next_persona_key(self) -> str:
        key = self._order[self._turn % len(self._order)]
        self._turn += 1
        return key

    def reset_rotation(self) -> None:
        """Start rotation from the beginning."""
        self._turn = 0

    def set_rotation(self, order: list[str]) -> None:
        """Optionally change the rotation order (e.g., ["tech","tech","mentor"])."""
        if not order:
            raise ValueError("Rotation order cannot be empty.")
        for k in order:
            if k not in self.personas:
                raise KeyError(f"Unknown persona key in rotation: {k}")
        self._order = order
        self._turn = 0

    def list_personas(self) -> list[str]:
        return list(self.personas.keys())

    # ----- API used by FastAPI endpoints -----
    def ask(self, context: str) -> Dict[str, str]:
        """
        Pick the next persona (by rotation) and have it ask one interview question.
        Returns: {"persona": <key>, "question": <text>}
        """
        key = self._next_persona_key()
        agent = self.personas[key]
        question = agent.ask(context)
        return {"persona": key, "question": question}

    def evaluate(self, persona: str, question: str, answer: str) -> Dict[str, str]:
        """
        Have the specified persona evaluate the answer.
        Returns: {"persona": <key>, "feedback": <text>}
        """
        if persona not in self.personas:
            raise KeyError(f"Unknown persona: {persona}")
        agent = self.personas[persona]
        feedback = agent.feedback(question, answer)
        return {"persona": persona, "feedback": feedback}
