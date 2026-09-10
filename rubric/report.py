"""Render a self-contained report.html from the collected run results.

No templating engine or JS: stdlib string building, html.escape for safety, and
native <details>/<summary> elements for collapsible per-prompt detail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from .heuristics import Verdict

if TYPE_CHECKING:  # avoid importing client (and openai) at runtime / in tests
    from .client import PromptResult

# Per-prompt / per-model statuses.
PASS = "PASS"
FAIL = "FAIL"   # genuine garbled output
WARN = "WARN"   # incomplete: truncated or empty (not garble)
ERROR = "ERROR"  # the call itself failed

# Model status precedence: a real garble finding is the headline, then untested
# (ERROR), then incomplete (WARN), then clean (PASS).
_PRECEDENCE = [FAIL, ERROR, WARN, PASS]

_STATUS_META = {
    PASS: ("✅", "pass"),
    FAIL: ("❌", "fail"),
    WARN: ("⚠️", "warn"),
    ERROR: ("🔌", "error"),
}


def classify(result: "PromptResult", verdict: Verdict | None) -> tuple[str, str | None]:
    """Map a call result + heuristic verdict to a (status, note) pair.

    Truncation and emptiness are operational issues (WARN), kept distinct from
    genuine garbled text (FAIL) so the headline status reflects the tool's purpose.
    """
    if result.is_error:
        return ERROR, result.error

    text = result.text or ""
    if not text.strip():
        if result.is_truncated:
            return WARN, "Truncated at the token limit before any content was produced (raise max_tokens)."
        return WARN, "Model returned an empty response."

    if verdict and not verdict.passed:
        return FAIL, None  # the heuristic flags carry the detail

    notes = []
    if result.reasoning_used:
        notes.append("Answer taken from reasoning_content (message.content was empty).")
    if result.is_truncated:
        notes.append("Response was truncated at the token limit (may be cut off mid-sentence).")
        return WARN, " ".join(notes)
    return PASS, " ".join(notes) or None


@dataclass
class PromptOutcome:
    prompt_id: str
    prompt_text: str
    result: "PromptResult"
    verdict: Verdict | None  # None when the call errored
    status: str = PASS
    note: str | None = None


@dataclass
class CapabilityOutcome:
    """Result of a capability probe (tool calling / structured output)."""

    name: str            # short id, e.g. "tool-calling"
    label: str           # human label shown in the report
    description: str     # what the probe asked the model to do
    status: str
    detail: str | None = None      # judge explanation
    body: str | None = None        # raw response / tool call to show in <pre>
    finish_reason: str | None = None
    temperature: float | None = None
    is_error: bool = False


@dataclass
class ModelOutcome:
    model: str
    prompts: list[PromptOutcome] = field(default_factory=list)
    capabilities: list[CapabilityOutcome] = field(default_factory=list)

    @property
    def status(self) -> str:
        present = {p.status for p in self.prompts} | {c.status for c in self.capabilities}
        for s in _PRECEDENCE:
            if s in present:
                return s
        return PASS


_CSS = """
:root { --pass:#1a7f37; --fail:#cf222e; --warn:#9a6700; --error:#57606a; --bg:#f6f8fa; --border:#d0d7de; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 0; padding: 2rem; color: #1f2328; background: #fff; }
h1 { margin: 0 0 .25rem; font-size: 1.6rem; }
.meta { color: #57606a; font-size: .9rem; margin-bottom: 1.5rem; }
.legend { color: #57606a; font-size: .82rem; margin: -1rem 0 1.5rem; }
.summary { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.pill { padding: .5rem 1rem; border-radius: 8px; font-weight: 600; border: 1px solid var(--border); }
.pill.pass { color: var(--pass); } .pill.fail { color: var(--fail); }
.pill.warn { color: var(--warn); } .pill.error { color: var(--error); }
.model { border: 1px solid var(--border); border-radius: 10px; margin-bottom: 1rem; overflow: hidden; }
.model > summary { list-style: none; cursor: pointer; padding: .9rem 1.1rem; background: var(--bg);
                   display: flex; align-items: center; gap: .6rem; font-weight: 600; font-size: 1.05rem; }
.model > summary::-webkit-details-marker { display: none; }
.badge { font-size: .8rem; font-weight: 700; padding: .15rem .55rem; border-radius: 999px; border: 1px solid var(--border); }
.badge.pass { color: var(--pass); } .badge.fail { color: var(--fail); }
.badge.warn { color: var(--warn); } .badge.error { color: var(--error); }
.section { border-top: 1px solid var(--border); padding: .6rem 1.1rem; background: #fbfcfd;
           font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: #57606a; }
.prompt { border-top: 1px solid var(--border); padding: .9rem 1.1rem; }
.prompt .q { font-weight: 600; margin: 0 0 .4rem; display: flex; gap: .5rem; align-items: baseline; flex-wrap: wrap; }
.prompt .tag { font-size: .8rem; color: #57606a; font-weight: 400; }
.chip { font-size: .72rem; font-weight: 700; padding: .05rem .45rem; border-radius: 999px; border: 1px solid var(--border); }
.chip.pass { color: var(--pass); } .chip.fail { color: var(--fail); }
.chip.warn { color: var(--warn); } .chip.error { color: var(--error); }
.finish { font-size: .75rem; color: #57606a; margin: .35rem 0 0; font-family: ui-monospace, monospace; }
.note { font-size: .82rem; color: var(--warn); margin: .35rem 0 0; }
pre { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
      padding: .7rem; white-space: pre-wrap; word-break: break-word; margin: .5rem 0 0; font-size: .9rem; }
.flags { margin: .6rem 0 0; padding: 0; list-style: none; }
.flags li { color: var(--fail); font-size: .88rem; margin: .2rem 0; }
.flags li code { background: #ffebe9; padding: .05rem .35rem; border-radius: 4px; }
.errmsg { color: var(--error); font-family: ui-monospace, monospace; font-size: .9rem; }
""".strip()


def _redact(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}…{key[-4:]}"


def _fmt_temp(t: float | None) -> str:
    if t is None:
        return "?"
    return str(int(t)) if float(t).is_integer() else str(t)


def _prompt_block(p: PromptOutcome) -> str:
    _, cls = _STATUS_META[p.status]
    parts = ['<div class="prompt">']
    parts.append(
        f'<p class="q"><span class="chip {cls}">{p.status}</span>'
        f'{escape(p.prompt_id)} <span class="tag">— {escape(p.prompt_text)}</span></p>'
    )

    if p.result.is_error:
        parts.append(f'<p class="errmsg">ERROR: {escape(p.result.error or "")}</p>')
    else:
        parts.append(f"<pre>{escape(p.result.text or '')}</pre>")
        if p.verdict and p.verdict.flags:
            parts.append('<ul class="flags">')
            for f in p.verdict.flags:
                parts.append(f"<li><code>{escape(f.check)}</code> {escape(f.reason)}</li>")
            parts.append("</ul>")
        parts.append(
            f'<p class="finish">finish_reason: {escape(str(p.result.finish_reason))}'
            f' · temperature: {escape(_fmt_temp(p.result.temperature))}</p>'
        )

    if p.note:
        parts.append(f'<p class="note">{escape(p.note)}</p>')
    parts.append("</div>")
    return "".join(parts)


def _capability_block(c: CapabilityOutcome) -> str:
    _, cls = _STATUS_META[c.status]
    parts = ['<div class="prompt">']
    parts.append(
        f'<p class="q"><span class="chip {cls}">{c.status}</span>'
        f'{escape(c.label)} <span class="tag">— {escape(c.description)}</span></p>'
    )
    if c.is_error:
        parts.append(f'<p class="errmsg">{escape(c.detail or "")}</p>')
    else:
        if c.body:
            parts.append(f"<pre>{escape(c.body)}</pre>")
        if c.detail:
            note_cls = "note" if c.status in (WARN, PASS) else "flags"
            if note_cls == "flags":
                parts.append(f'<ul class="flags"><li>{escape(c.detail)}</li></ul>')
            else:
                parts.append(f'<p class="note">{escape(c.detail)}</p>')
        if c.finish_reason is not None or c.temperature is not None:
            parts.append(
                f'<p class="finish">finish_reason: {escape(str(c.finish_reason))}'
                f' · temperature: {escape(_fmt_temp(c.temperature))}</p>'
            )
    parts.append("</div>")
    return "".join(parts)


def _model_block(m: ModelOutcome) -> str:
    icon, cls = _STATUS_META[m.status]
    prompts_html = "".join(_prompt_block(p) for p in m.prompts)
    caps_html = ""
    if m.capabilities:
        caps_html = '<div class="section">Capability checks</div>' + "".join(
            _capability_block(c) for c in m.capabilities
        )
    open_attr = "" if m.status == PASS else " open"
    return (
        f'<details class="model"{open_attr}>'
        f'<summary>{icon} {escape(m.model)} '
        f'<span class="badge {cls}">{m.status}</span></summary>'
        f"{prompts_html}{caps_html}"
        f"</details>"
    )


def build_report(
    outcomes: list[ModelOutcome],
    *,
    timestamp: str,
    api_key: str,
    output_path: str | Path = "report.html",
) -> Path:
    """Write report.html and return its path."""
    counts = {PASS: 0, FAIL: 0, WARN: 0, ERROR: 0}
    for m in outcomes:
        counts[m.status] += 1

    summary = "".join(
        f'<span class="pill {_STATUS_META[s][1]}">{_STATUS_META[s][0]} {counts[s]} {s}</span>'
        for s in (PASS, FAIL, WARN, ERROR)
    )
    models_html = "".join(_model_block(m) for m in outcomes)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DO Serverless Inference — garbled-output report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>DO Serverless Inference — garbled-output report</h1>
<p class="meta">Generated {escape(timestamp)} · endpoint <code>https://inference.do-ai.run/v1/</code> · key <code>{escape(_redact(api_key))}</code> · {len(outcomes)} model(s)</p>
<p class="legend"><b>PASS</b> coherent English · <b>FAIL</b> garbled output detected · <b>WARN</b> incomplete (truncated or empty, not garble) · <b>ERROR</b> request failed</p>
<div class="summary">{summary}</div>
{models_html}
</body>
</html>
"""
    path = Path(output_path)
    path.write_text(html, encoding="utf-8")
    return path
