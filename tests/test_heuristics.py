"""Network-free sanity checks for the garbled-text heuristics.

Run with:  python -m pytest    (or)    python tests/test_heuristics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rubric.heuristics import evaluate  # noqa: E402
from rubric.report import PASS, FAIL, WARN, ERROR, classify  # noqa: E402
from rubric.capabilities import evaluate_tool_call, evaluate_structured_output  # noqa: E402


class _Cap:
    """Duck-typed stand-in for client.CapabilityResult."""

    def __init__(self, text=None, tool_calls=None, error=None, unsupported=False):
        self.text = text
        self.tool_calls = tool_calls
        self.error = error
        self.unsupported = unsupported

    @property
    def is_error(self):
        return self.error is not None


def _tc(name="get_weather", arguments='{"location": "Paris, France"}'):
    return [{"name": name, "arguments": arguments}]


class _R:
    """Lightweight stand-in for client.PromptResult (avoids importing openai)."""

    def __init__(self, text=None, error=None, finish_reason="stop", reasoning_used=False):
        self.text = text
        self.error = error
        self.finish_reason = finish_reason
        self.reasoning_used = reasoning_used
        self.temperature = 0.0

    @property
    def is_error(self):
        return self.error is not None

    @property
    def is_truncated(self):
        return self.finish_reason == "length"


def _classify(result):
    verdict = None if result.is_error else evaluate(result.text or "")
    return classify(result, verdict)


def _flags(text):
    return {f.check for f in evaluate(text).flags}


def test_clean_english_passes():
    text = (
        "The capital of France is Paris. It is one of the most visited cities "
        "in the world, known for its art, food, and history."
    )
    v = evaluate(text)
    assert v.passed, v.flags


def test_short_valid_answer_passes():
    assert evaluate("Paris.").passed


def test_empty_fails():
    assert "empty" in _flags("")
    assert "empty" in _flags("   \n\t ")


def test_repetition_loop_fails():
    assert "repetition" in _flags("the the the the the the the the the the")


def test_consonant_soup_fails():
    assert "wordlike" in _flags("bcdfg hjklm npqrs tvwxz bcdfg hjklm npqrs")


def test_non_ascii_flood_fails():
    assert "non_ascii" in _flags("這是一段中文文字這是一段中文文字這是一段中文文字")


def test_mojibake_symbols_fail():
    flags = _flags("�����������������������������")
    assert "symbols" in flags or "wordlike" in flags


def test_classify_clean_passes():
    assert _classify(_R(text="The sky is blue because of Rayleigh scattering."))[0] == PASS


def test_classify_truncated_warns():
    status, note = _classify(_R(text="A coherent answer cut off mid", finish_reason="length"))
    assert status == WARN and "truncated" in note.lower()


def test_classify_empty_warns_not_fails():
    assert _classify(_R(text="", finish_reason="stop"))[0] == WARN
    assert _classify(_R(text="", finish_reason="length"))[0] == WARN


def test_classify_reasoning_fallback_passes_with_note():
    status, note = _classify(_R(text="The real answer lives here.", reasoning_used=True))
    assert status == PASS and "reasoning_content" in note


def test_classify_garbled_fails():
    assert _classify(_R(text="bcdfg hjklm npqrs tvwxz bcdfg hjklm"))[0] == FAIL


def test_classify_error():
    assert _classify(_R(error="AuthenticationError: 401"))[0] == ERROR


def test_tool_call_correct_passes():
    assert evaluate_tool_call(_Cap(tool_calls=_tc()))[0] == PASS


def test_tool_call_missing_fails():
    assert evaluate_tool_call(_Cap(tool_calls=[]))[0] == FAIL


def test_tool_call_wrong_name_fails():
    assert evaluate_tool_call(_Cap(tool_calls=_tc(name="lookup")))[0] == FAIL


def test_tool_call_bad_json_args_fails():
    assert evaluate_tool_call(_Cap(tool_calls=_tc(arguments="{not json")))[0] == FAIL


def test_tool_call_missing_location_fails():
    assert evaluate_tool_call(_Cap(tool_calls=_tc(arguments='{"unit": "celsius"}')))[0] == FAIL


def test_tool_call_unsupported_warns():
    assert evaluate_tool_call(_Cap(error="400 tools not supported", unsupported=True))[0] == WARN


def test_tool_call_other_error():
    assert evaluate_tool_call(_Cap(error="AuthenticationError: 401"))[0] == ERROR


def test_structured_valid_passes():
    ok = _Cap(text='{"name": "Ada Lovelace", "age": 36, "city": "London"}')
    assert evaluate_structured_output(ok)[0] == PASS


def test_structured_bad_json_fails():
    assert evaluate_structured_output(_Cap(text="not json at all"))[0] == FAIL


def test_structured_missing_key_fails():
    assert evaluate_structured_output(_Cap(text='{"name": "Ada", "city": "London"}'))[0] == FAIL


def test_structured_wrong_type_fails():
    assert evaluate_structured_output(_Cap(text='{"name": "Ada", "age": "36", "city": "London"}'))[0] == FAIL


def test_structured_unsupported_warns():
    cap = _Cap(error="400 response_format json_schema not supported", unsupported=True)
    assert evaluate_structured_output(cap)[0] == WARN


def _run_all():
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
            passed += 1
    print(f"\n{passed} passed")


if __name__ == "__main__":
    _run_all()
