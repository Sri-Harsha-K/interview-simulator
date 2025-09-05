from __future__ import annotations
from typing import Dict
from .base import BaseAgent, PersonaConfig

# --- Persona config ---------------------------------------------------------
TECH_CFG = PersonaConfig(
    name="Senior Data Science Interviewer",
    system_prompt=(
        "You are a senior data science interviewer."
        "Focus on: problem framing, EDA, feature engineering, modeling, evaluation, experimentation, causal inference, time series, recommendation, NLP, ML ops/monitoring, fairness, and cost/latency trade-offs."
        "Ask concrete, scenario-driven questions with realistic constraints and numbers."
        "Write in direct, plain English."
    ),
)


class TechAgent(BaseAgent):
    """Senior Data Science interviewer with domain-specific helpers.

    Backwards compatible with BaseAgent: keeps the same ask() and feedback() surface, and
    adds optional helpers that you *may* call from the app/orchestrator if desired.

    Hints in context (all optional, case-insensitive):
      - Difficulty tag: [easy] | [medium] | [hard]
      - topic:<topic>        (e.g., topic:ab-testing, topic:causal, topic:time-series)
      - stack:<lang>         (e.g., stack:python, stack:r)
      - framework:<lib>      (e.g., framework:sklearn, framework:xgboost, framework:pytorch)
      - mlops:<tool>         (e.g., mlops:mlflow, mlops:sagemaker)
      - domain:<area>        (e.g., domain:ads, domain:marketplace, domain:logistics, domain:healthcare)
      - data:<shape>         (e.g., data:class-imbalance, data:10m-rows)
    """

    # --- Question generation ------------------------------------------------
    def ask(self, context: str) -> str:
        """Ask one concrete data-science interview question, honoring simple hints in context."""
        difficulty = self._extract_difficulty(context)
        hints = self._extract_hints(context)
        hint_line = f"Difficulty: {difficulty}." + (f" Hints: {', '.join(hints)}." if hints else "")

        prompt = (
            "Ask exactly ONE data-science interview question that is concrete and scenario-based."
            "Target <30 words. No preface, no bullets, no numbering."
            "Choose ONE topic: problem framing, EDA, leakage, feature engineering, model selection, cross-validation, hyperparameter tuning, regularization, class imbalance,"
            "evaluation metrics (AUC/PR-AUC/RMSE/MAE/logloss/recall@k), calibration, A/B testing (power/variance reduction), causal inference (DID/IV/PSM),"
            "time series (stationarity/seasonality/forecasting), recommendation (cold start/offline vs online metrics), NLP (embeddings/transformers/RAG eval),"
            "model monitoring (drift/perf), fairness, or cost/latency trade-offs."
            f"{hint_line}"
            f"Candidate/context:{context}"
        )
        return self.llm.complete_chat(self._messages(prompt), max_tokens=128, temperature=0.6).strip()

    # --- Domain-specific helpers (optional) ---------------------------------
    def followup(self, question: str, answer: str) -> str:
        """Return ONE probing follow-up targeting the weakest area (≤20 words)."""
        prompt = (
            "Based on the candidate's answer, ask ONE probing follow-up targeting the weakest area."
            "≤20 words, concrete; ask for a number, formula, threshold, specific test, or trade-off. No preface."
            f"Original question: {question}"
            f"Candidate answer: {answer}"
        )
        return self.llm.complete_chat(self._messages(prompt), max_tokens=80, temperature=0.5).strip()

    def score(self, question: str, answer: str) -> Dict[str, int]:
        """
        Produce a compact rubric (1–5) for: depth, correctness, tradeoffs, communication, with a ≤20-word summary.
        Returns a dict; if parsing fails, fields default to 0 and summary contains the raw string.
        """
        prompt = (
            "Score the answer on 1–5 for: depth, correctness, tradeoffs, communication."
            "Return ONLY compact JSON with keys: depth, correctness, tradeoffs, communication, summary."
            "Summary ≤20 words. No other text."
            f"Question: {question} Answer: {answer}"
        )
        raw = self.llm.complete_chat(self._messages(prompt), max_tokens=120, temperature=0.0).strip()
        try:
            import json
            data = json.loads(raw)
            for k in ("depth", "correctness", "tradeoffs", "communication", "summary"):
                data.setdefault(k, 0 if k != "summary" else "")
            return data
        except Exception:
            return {"depth": 0, "correctness": 0, "tradeoffs": 0, "communication": 0, "summary": raw[:120]}

    # --- small utilities -----------------------------------------------------
    @staticmethod
    def _extract_difficulty(context: str) -> str:
        lc = context.lower()
        if "[easy]" in lc:
            return "easy"
        if "[hard]" in lc:
            return "hard"
        return "medium"

    @staticmethod
    def _extract_hints(context: str) -> list[str]:
        lc = context.lower()
        found: list[str] = []
        for key in ("topic", "stack", "framework", "mlops", "domain", "data"):
            token = f"{key}:"
            if token in lc:
                try:
                    seg = lc.split(token, 1)[1].split()[0].strip(",;.")
                    if seg:
                        found.append(f"{key}={seg}")
                except Exception:
                    pass
        return found
