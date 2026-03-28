"""
Base class for all agents in the survey system.

Provides LLM calling (via LLMClient) and JSON response parsing.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from llm_client import LLMClient
from state import PaperState


class BaseAgent(ABC):
    """Abstract base for every agent. Subclasses implement `run`."""

    def __init__(
        self,
        name: str,
        client: LLMClient,
        system_prompt: str,
    ):
        self.name = name
        self.client = client
        self.system_prompt = system_prompt
        self.conversation_history: list[dict] = []

    # ── LLM Calls ────────────────────────────────────────────────────

    def _call_llm(
        self,
        user_message: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Multi-turn call — accumulates conversation history."""
        self.conversation_history.append({"role": "user", "content": user_message})
        text = self.client.chat(
            messages=self.conversation_history,
            system=self.system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.conversation_history.append({"role": "assistant", "content": text})
        return text

    def _call_llm_once(
        self,
        user_message: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Stateless single-shot call — no history accumulation."""
        return self.client.chat(
            messages=[{"role": "user", "content": user_message}],
            system=self.system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def reset_history(self) -> None:
        """Clear conversation history between phases."""
        self.conversation_history = []

    # ── JSON Parsing ─────────────────────────────────────────────────

    @staticmethod
    def parse_json(text: str) -> dict | list:
        """
        Extract and parse JSON from an LLM response.

        Handles common issues:
        - JSON wrapped in ```json ... ``` code fences
        - Leading/trailing non-JSON text
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences
        fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON array or object in the text
        for pattern in [r"\[.*\]", r"\{.*\}"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not parse JSON from response:\n{text[:500]}")

    def _call_llm_json(
        self,
        user_message: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> dict | list:
        """Call LLM and parse JSON response, with one retry on parse failure."""
        text = self._call_llm_once(user_message, max_tokens, temperature)
        try:
            return self.parse_json(text)
        except ValueError:
            retry_msg = (
                "Your previous response was not valid JSON. "
                "Please respond with ONLY a valid JSON object/array, "
                "no other text or markdown formatting."
            )
            text = self._call_llm_once(
                f"{retry_msg}\n\nOriginal request:\n{user_message}",
                max_tokens,
                temperature,
            )
            return self.parse_json(text)

    # ── Abstract Interface ───────────────────────────────────────────

    @abstractmethod
    def run(self, state: PaperState) -> PaperState:
        """Execute this agent's task. Reads from and writes to state."""
        ...
