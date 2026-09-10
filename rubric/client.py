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


@dataclass
class CapabilityResult:
    """Outcome of a tool-calling or structured-output probe.

    ``unsupported`` marks a request the model's endpoint rejected specifically
    because the feature isn't available (reported as WARN, not a hard ERROR).
    """

    text: str | None = None
    tool_calls: list[dict] | None = None
    error: str | None = None
    unsupported: bool = False
    finish_reason: str | None = None
    temperature: float | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


# Hints used to tell "the model doesn't support this feature" apart from a
# generic request failure (auth, timeout, unknown model, …).
_UNSUPPORTED_HINTS = (
    "not support",
    "unsupported",
    "not available",
    "not enabled",
    "no support",
    "does not accept",
    "is not allowed",
    "invalid parameter",
    "invalid_request",
    "400",
)


def _is_unsupported(exc: Exception, feature_keywords: tuple[str, ...]) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in feature_keywords) and any(h in msg for h in _UNSUPPORTED_HINTS)


def _attempt(do_call):
    """Run ``do_call(temperature)``; if rejected for the temperature value, retry
    once at the fallback temperature. Returns (completion, temperature, exception)."""
    temperature = DEFAULT_TEMPERATURE
    try:
        return do_call(temperature), temperature, None
    except Exception as exc:  # noqa: BLE001
        if "temperature" in str(exc).lower():
            temperature = FALLBACK_TEMPERATURE
            try:
                return do_call(temperature), temperature, None
            except Exception as retry_exc:  # noqa: BLE001
                return None, temperature, retry_exc
        return None, temperature, exc


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
    completion, temperature, exc = _attempt(lambda t: _create(client, model, prompt_text, t))
    if exc is not None:
        return PromptResult(error=f"{type(exc).__name__}: {exc}", temperature=temperature)

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


# --- Capability probes: tool calling and structured output ------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. 'Paris, France'.",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}
TOOL_PROMPT = (
    "What is the current temperature in Paris, France? "
    "Use the get_weather tool to look it up."
)
_TOOL_KEYWORDS = ("tool", "tools", "tool_choice", "function call", "function_call")

PERSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "person",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "city"],
            "additionalProperties": False,
        },
    },
}
STRUCT_PROMPT = (
    "Extract the person's details from this sentence and return them as JSON: "
    "'Ada Lovelace is 36 years old and lives in London.'"
)
_STRUCT_KEYWORDS = ("response_format", "json_schema", "json schema", "structured output")


def run_tool_call(client: OpenAI, model: str) -> CapabilityResult:
    """Probe whether the model emits a correct tool call. Never raises."""

    def do(t):
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": TOOL_PROMPT}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            temperature=t,
            max_tokens=MAX_TOKENS,
        )

    completion, temperature, exc = _attempt(do)
    if exc is not None:
        return CapabilityResult(
            error=f"{type(exc).__name__}: {exc}",
            unsupported=_is_unsupported(exc, _TOOL_KEYWORDS),
            temperature=temperature,
        )

    choice = completion.choices[0]
    raw_calls = choice.message.tool_calls or []
    calls = [
        {"name": c.function.name, "arguments": c.function.arguments}
        for c in raw_calls
        if getattr(c, "function", None) is not None
    ]
    return CapabilityResult(
        tool_calls=calls,
        finish_reason=choice.finish_reason,
        temperature=temperature,
    )


def run_structured_output(client: OpenAI, model: str) -> CapabilityResult:
    """Probe whether the model returns JSON conforming to a strict schema. Never raises."""

    def do(t):
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": STRUCT_PROMPT}],
            response_format=PERSON_SCHEMA,
            temperature=t,
            max_tokens=MAX_TOKENS,
        )

    completion, temperature, exc = _attempt(do)
    if exc is not None:
        return CapabilityResult(
            error=f"{type(exc).__name__}: {exc}",
            unsupported=_is_unsupported(exc, _STRUCT_KEYWORDS),
            temperature=temperature,
        )

    choice = completion.choices[0]
    text, _ = _extract_content(choice.message)
    return CapabilityResult(
        text=text or "",
        finish_reason=choice.finish_reason,
        temperature=temperature,
    )
