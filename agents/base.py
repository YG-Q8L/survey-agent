"""
Base class for all agents in the survey system.

Provides LLM calling, retry logic, and JSON response parsing.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod

from anthropic import Anthropic, APIError, RateLimitError

from state import PaperState


class BaseAgent(ABC):
    """Abstract base for every agent. Subclasses implement `run`."""

    def __init__(
        self,
        name: str,
        client: Anthropic,
        model: str,
        system_prompt: str,
    ):
        self.name = name
        self.client = client
        self.model = model
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
        text = self._do_api_call(self.conversation_history, max_tokens, temperature)
        self.conversation_history.append({"role": "assistant", "content": text})
        return text

    def _call_llm_once(
        self,
        user_message: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Stateless single-shot call — no history accumulation."""
        messages = [{"role": "user", "content": user_message}]
        return self._do_api_call(messages, max_tokens, temperature)

    def _do_api_call(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        retries: int = 3,
    ) -> str:
        """Actual API call with retry on rate-limit / transient errors."""
        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=self.system_prompt,
                    messages=messages,
                )
                return response.content[0].text
            except RateLimitError:
                wait = 2**attempt * 5  # 5s, 10s, 20s
                print(f"[{self.name}] Rate limited, waiting {wait}s...")
                time.sleep(wait)
            except APIError as e:
                print(f"[{self.name}] API error (attempt {attempt + 1}): {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(2)
        raise RuntimeError(f"Agent {self.name} failed after {retries} retries")

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

        raise ValueError(f"[{__class__.__name__}] Could not parse JSON from response:\n{text[:500]}")

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
            # Retry with explicit instruction
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
