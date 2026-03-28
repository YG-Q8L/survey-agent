"""
Unified LLM client — abstracts Anthropic and OpenRouter (OpenAI-compatible) APIs.

Usage:
    client = create_client("anthropic", model="claude-sonnet-4-20250514")
    client = create_client("openrouter", model="anthropic/claude-sonnet-4")

Agents call client.chat() and get a string back, regardless of provider.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass
class LLMClient:
    """Thin wrapper that gives agents a uniform interface."""

    provider: str   # "anthropic" | "openrouter"
    model: str
    _client: object = None

    def chat(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        retries: int = 3,
    ) -> str:
        """Send messages and return the assistant's text response."""
        for attempt in range(retries):
            try:
                if self.provider == "anthropic":
                    return self._chat_anthropic(messages, system, max_tokens, temperature)
                else:
                    return self._chat_openai_compat(messages, system, max_tokens, temperature)
            except Exception as e:
                is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
                if is_rate_limit:
                    wait = 2**attempt * 5
                    print(f"[LLMClient] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                elif attempt == retries - 1:
                    raise
                else:
                    print(f"[LLMClient] Error (attempt {attempt + 1}): {e}")
                    time.sleep(2)
        raise RuntimeError(f"LLM call failed after {retries} retries")

    def _chat_anthropic(
        self, messages: list[dict], system: str, max_tokens: int, temperature: float,
    ) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    def _chat_openai_compat(
        self, messages: list[dict], system: str, max_tokens: int, temperature: float,
    ) -> str:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=full_messages,
        )
        return response.choices[0].message.content


def create_client(provider: str, model: str) -> LLMClient:
    """
    Factory function — creates the appropriate SDK client.

    Anthropic: reads ANTHROPIC_API_KEY
    OpenRouter: reads OPENROUTER_API_KEY
    """
    if provider == "anthropic":
        from anthropic import Anthropic
        client = LLMClient(provider=provider, model=model)
        client._client = Anthropic()  # reads ANTHROPIC_API_KEY from env
        return client

    elif provider == "openrouter":
        from openai import OpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        client = LLMClient(provider=provider, model=model)
        client._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        return client

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openrouter'.")
