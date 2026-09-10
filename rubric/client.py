"""Thin wrapper around the OpenAI SDK pointed at DigitalOcean Serverless Inference."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

BASE_URL = "https://inference.do-ai.run/v1/"
MAX_TOKENS = 4096
TIMEOUT_SECONDS = 60.0

# Preferred sampling temperature (deterministic so runs are comparable). Some
# models reject anything other than 1 — we detect that and retry at 1.0.
DEFAULT_TEMPERATURE = 0.0
FALLBACK_TEMPERATURE = 1.0


@dataclass
class PromptResult:
    """Outcome of a single (model, prompt) call.

    On success ``text`` holds the answer and ``error`` is None; on failure the
    reverse. ``finish_reason`` and ``reasoning_used`` explain *how* the text was
    obtained so the report can distinguish a truncated/empty reply from a real one.
    """

    text: str | None = None
    error: str | None = None
    finish_reason: str | None = None
    reasoning_used: bool = False
    temperature: float | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def is_truncated(self) -> bool:
        return self.finish_reason == "length"


def make_client(api_key: str) -> OpenAI:
    """Build an OpenAI client that talks to DO's OpenAI-compatible endpoint."""
    return OpenAI(base_url=BASE_URL, api_key=api_key, timeout=TIMEOUT_SECONDS)


def _create(client: OpenAI, model: str, prompt_text: str, temperature: float):
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )


def _extract_content(message) -> tuple[str, bool]:
    """Return (text, reasoning_used). Falls back to reasoning_content when the
    normal content is empty — some reasoning models put the answer there."""
    content = (message.content or "").strip() if message.content is not None else ""
    if content:
        return message.content, False

    # Reasoning models may expose the answer under a separate field.
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None:
        extra = getattr(message, "model_extra", None)
        if extra:
            reasoning = extra.get("reasoning_content")
    if reasoning and reasoning.strip():
        return reasoning, True

    return message.content or "", False


def run_prompt(client: OpenAI, model: str, prompt_text: str) -> PromptResult:
    """Send one prompt to one model. Never raises — errors are captured in the result.

    Some DO models (e.g. kimi-k2.6) only accept ``temperature=1``. If the first
    attempt is rejected specifically because of the temperature value, retry once
    at the fallback temperature rather than reporting a spurious ERROR.
    """
    temperature = DEFAULT_TEMPERATURE
    try:
        completion = _create(client, model, prompt_text, temperature)
    except Exception as exc:  # noqa: BLE001 — one bad model must not abort the run
        if "temperature" in str(exc).lower():
            temperature = FALLBACK_TEMPERATURE
            try:
                completion = _create(client, model, prompt_text, temperature)
            except Exception as retry_exc:  # noqa: BLE001
                return PromptResult(error=f"{type(retry_exc).__name__}: {retry_exc}")
        else:
            return PromptResult(error=f"{type(exc).__name__}: {exc}")

    if not completion.choices:
        return PromptResult(error="Response contained no choices.", temperature=temperature)

    choice = completion.choices[0]
    text, reasoning_used = _extract_content(choice.message)
    return PromptResult(
        text=text or "",
        finish_reason=choice.finish_reason,
        reasoning_used=reasoning_used,
        temperature=temperature,
    )
