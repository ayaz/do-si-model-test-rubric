# DO Serverless Inference — garbled-output checker

A small Python tool that probes a configured list of open-weights models served
through [DigitalOcean Serverless Inference](https://docs.digitalocean.com/products/inference/how-to/si-endpoints/)
and reports whether any of them returns **garbled / incoherent text** in response
to plain English prompts. Each run writes a self-contained `report.html`.

## How it works

1. Reads the model list (and optional detection thresholds) from `models.yaml`.
2. For each model, sends a small built-in set of English prompts via the
   OpenAI-compatible endpoint `https://inference.do-ai.run/v1/`.
3. Runs deterministic, network-free **heuristics** over each response
   (non-ASCII flood, symbol/mojibake noise, repetition loops,
   consonant-soup gibberish).
4. Writes `report.html`: per-model status with expandable per-prompt responses,
   the reason any heuristic flagged them, and each call's `finish_reason` /
   temperature.

Detection is heuristic-only — no second "judge" model, no extra API calls.

### Statuses

| Status | Meaning |
| --- | --- |
| ✅ **PASS** | Coherent English. |
| ❌ **FAIL** | Genuine garbled output detected by a heuristic. |
| ⚠️ **WARN** | Incomplete, *not* garble — response was truncated at the token limit or came back empty. |
| 🔌 **ERROR** | The request itself failed (auth, unknown model, etc.). |

Truncation and emptiness are treated as operational issues (WARN), kept separate
from garbled text (FAIL), so a model is only marked FAIL when it actually produces
incoherent text. Model status precedence is FAIL → ERROR → WARN → PASS.

Requests are sent at `temperature=0` for reproducible runs; models that only
accept `temperature=1` (e.g. `kimi-k2.6`) are retried automatically at 1.0. When
a model returns empty `content` but provides `reasoning_content`, the answer is
read from there.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit .env and set your key
```

Set your DO **model access key** (create one in the DigitalOcean control panel):

```bash
export DIGITALOCEAN_INFERENCE_KEY="your-model-access-key"
# ...or put it in .env — run.py loads .env automatically.
```

Edit `models.yaml` and list the exact model IDs you want to check (as shown in
the DO control panel / the `GET /v1/models` endpoint).

## Run

```bash
python run.py
```

Then open `report.html` in a browser. Options:

```bash
python run.py --config models.yaml --output report.html
```

## Tuning detection

Override any threshold under `thresholds:` in `models.yaml`; omitted keys fall
back to the defaults in `rubric/heuristics.py`:

| Key | Meaning | Default |
| --- | --- | --- |
| `max_non_ascii_ratio` | max fraction of non-ASCII letters | 0.15 |
| `max_symbol_ratio` | max fraction of noise/symbol chars | 0.30 |
| `max_repetition_ratio` | max share of the single most-frequent token | 0.50 |
| `min_wordlike_ratio` | min fraction of tokens that look like real words | 0.60 |

## Tests

Heuristics have network-free unit checks:

```bash
python -m pytest            # if pytest is installed
python tests/test_heuristics.py   # no pytest required
```

## Layout

```
run.py                 # entry point
models.yaml            # models to check + optional thresholds
rubric/config.py       # load config + API key
rubric/prompts.py      # built-in English prompt set
rubric/client.py       # OpenAI client for DO + per-prompt call
rubric/heuristics.py   # garbled-text detection
rubric/report.py       # report.html generation
tests/                 # heuristic sanity checks
```
