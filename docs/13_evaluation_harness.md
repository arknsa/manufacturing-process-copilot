# Document 13 — Evaluation Harness

**Status:** Active
**Script:** `backend/scripts/eval_copilot.py`
**Purpose:** make the reliability claims in [docs/11_evaluation.md](docs/11_evaluation.md)
reproducible from the repository.

---

## 1. What it is

`eval_copilot.py` is a self-contained reliability harness that reproduces the
runtime evaluation methodology from Document 11. It sends live chat requests for a
fixed set of supervisor intents, classifies each response, and reports aggregate
reliability metrics — optionally enriched by correlating the backend's structured
logs to recover tool-dispatch and iteration data exactly as Document 11 did.

It uses the **Python 3.10+ standard library only** (no third-party packages) and
**does not modify** agent behavior, prompts, or model configuration. It is a pure
observer of the running system.

---

## 2. Prerequisites

- The stack is running and healthy:
  ```bash
  docker compose up -d --build
  curl http://localhost:8000/ready    # {"status":"ready","ml_service_loaded":true}
  ```
- The database is seeded (`backend/scripts/seed_db.py`) so intents resolve to real
  data. See LIMITATIONS for how empty data affects the hallucination metric.
- For tool-dispatch and iteration metrics, the backend container name is known
  (default Compose name: `manufacturing-process-copilot-backend-1`).

---

## 3. How to run

### Quick check (client-observable metrics only)

```bash
python backend/scripts/eval_copilot.py --runs 10
```

Produces success rate, fallback rate, latency, and a per-intent breakdown for the
8 default intents (80 requests). No backend access beyond the HTTP API is needed.

### Full reproduction of docs/11 (with log correlation)

```bash
python backend/scripts/eval_copilot.py \
    --runs 10 \
    --docker-container manufacturing-process-copilot-backend-1 \
    --json-out eval_results.json \
    --csv-out eval_results.csv
```

This adds **tool dispatch rate**, **average iterations**, and the precise
**hallucination rate** (answers produced with no tool call), reproducing the
80-run table in Document 11.

### Useful flags

| Flag | Default | Purpose |
|---|---|---|
| `--base-url` | `http://localhost:8000` | Target backend |
| `--runs` | `10` | Runs per intent (matches docs/11) |
| `--intents-file` | — | Text file, one intent per line (overrides the 8 defaults) |
| `--delay` | `1.5` | Seconds between requests (keeps log turns non-interleaved) |
| `--timeout` | `120` | Per-request timeout (LLM calls can take 15–20s) |
| `--docker-container` | — | Enables log-derived metrics |
| `--json-out` / `--csv-out` | — | Write full results |

### Single-intent debugging

```bash
printf 'What orders are at high risk right now?\n' > intent.txt
python backend/scripts/eval_copilot.py --runs 3 --intents-file intent.txt \
    --docker-container manufacturing-process-copilot-backend-1
```

---

## 4. Expected output

A console summary, for example (verified run, 2 requests, log-correlated):

```
======================================================================
COPILOT RELIABILITY SUMMARY
======================================================================
Total runs:              2
Success rate:            100.0%
Fallback rate:           0.0%
Error rate:              0.0%
Latency p50 / p95 (ms):  11141.1 / 14383.8
Tool dispatch rate:      100.0%
Avg iterations:          2
Hallucination rate:      0.0%  (answered without a tool call)
(log-correlated runs:    2/2)
----------------------------------------------------------------------
Per-intent:
  [0] What orders are at high risk right now?    n=2 ok=100% fb=0% iter=2
======================================================================
```

**JSON output** (`--json-out`) contains `meta` (timestamp, config), `summary`
(aggregate + per-intent), and `runs` (every per-request record including the
tools dispatched and final iteration). **CSV output** (`--csv-out`) is one row per
run for spreadsheet analysis.

For the full 8-intent × 10-run reproduction, expect intents 1–6 at ~100% success
in ~2 iterations, and intents 7–8 dependent on whether a prediction row exists for
the demo order (see Document 11 §5).

---

## 5. Metric definitions and how they map to docs/11

| Harness metric | Definition | docs/11 equivalent |
|---|---|---|
| **success rate** | non-fallback, non-error, non-empty answer | the "OK" classification |
| **fallback rate** | answer begins with the agent's `_FALLBACK` string | "Fallback responses" |
| **tool dispatch rate** | runs with ≥1 `dispatching tool` log line | "80/80 runs dispatched a tool" |
| **avg iterations** | mean of `FINAL_ANSWER reached at iteration N` | "Avg iterations to answer" |
| **hallucination rate** | substantive answer with **0** tool dispatches (log-derived) | "fabricated production data" / "answered without tool" |
| **novel-order-id rate** | answer contains order IDs not in the prompt (heuristic) | weak proxy — see LIMITATIONS |
| **latency p50/p95** | wall-clock per request | API timing (complements docs/09) |

**Classification anchors** are taken directly from the code so the harness stays in
sync with the system:
- fallback string ← `agent.py :: _FALLBACK`
- log markers ← `agent.py` (`compress start for session`, `dispatching tool`,
  `FINAL_ANSWER reached at iteration`).

---

## 6. How it works

1. **Sequential live requests.** For each intent, `--runs` requests are sent with a
   fresh session token and a delay between them. Sequential issue (not concurrent)
   ensures backend log turns appear in the same order as the runs, so they can be
   segmented reliably.
2. **Client classification.** Each response is classified from the HTTP body alone
   (fallback / error / success, order-id heuristic, latency).
3. **Log correlation (optional).** When `--docker-container` is given, the harness
   reads `docker logs --since <elapsed+60s>`, segments lines by the per-turn
   `compress start for session <prefix>` marker, and attaches the dispatched tools
   and final iteration to each run by 8-char session prefix — the same correlation
   technique used to produce the Document 11 results.

---

## 7. Limitations

- **Hallucination precision requires logs.** The strong, precise signal
  ("substantive answer with zero tool calls") is only available with
  `--docker-container`. Without it, the harness reports the weaker
  **novel-order-id** heuristic, which is **not reliable on a seeded database**:
  when real high-risk orders exist, legitimate order IDs appear in answers and the
  heuristic over-counts. Treat the heuristic as a smoke alarm, not a measurement;
  use log correlation for the real number.
- **Log correlation is sequential-only.** It assumes runs do not interleave in the
  log stream. The default `--delay` enforces this. Running the backend under
  concurrent external traffic during evaluation will corrupt the segmentation.
- **Docker-specific log access.** Log enrichment shells out to `docker logs`. For
  non-Docker deployments, point the same parsing logic at your log source, or rely
  on the client-observable metrics.
- **LLM non-determinism.** Free-tier model output varies per call; this is why the
  default is 10 runs per intent. Small `--runs` values give noisy rates.
- **Data dependence.** Intents that require seeded rows (e.g. a prediction for the
  demo order) will legitimately return "no data available" answers if the database
  lacks them. That is correct behavior, not a failure — but it lowers the success
  rate for those intents. Seed appropriately before drawing conclusions.
- **Not yet a CI gate.** The harness is runnable on demand. Promoting it to an
  automated regression check (with thresholds) is listed as future work in
  Document 11 §8.

---

## 8. Assumptions

- The backend exposes `POST /api/v1/chat/message` with a `stream=false` JSON mode
  returning `{"content": "..."}` (as implemented in `api/routes/chat.py`).
- The agent's fallback string and log markers match the current `agent.py`. If
  those strings change, update the anchors at the top of `eval_copilot.py`
  (`FALLBACK_PREFIX`, `LOG_*_RE`).
- Order IDs follow the `ORD-YYYYMMDD-NNN` shape used by the seed data.

---

*Document 13 — Manufacturing Process Copilot Technical Series*
