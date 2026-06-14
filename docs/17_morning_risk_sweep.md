# Document 17 — W2: Morning Risk Sweep

**Status:** Implemented (importable n8n workflow)
**Artifact:** [`n8n/workflows/morning-risk-sweep.json`](../n8n/workflows/morning-risk-sweep.json)
**Architecture reference:** [docs/16_automation_architecture.md](16_automation_architecture.md) — Workflow **W2**

---

## 1. Business objective

Every production day starts with a question the shift supervisor answers manually:
*"Which of today's orders are most likely to slip, and why?"* Answering it by hand
means opening each order, reading the model output, and mentally ranking them — minutes
of work, done inconsistently, and easy to skip on a busy morning.

**The Morning Risk Sweep automates that first hour of triage.** At the start of each day
it scans every scheduled order, scores each one with the existing delay model, ranks them
by risk, and posts a single prioritized digest to the operations channel — before anyone
walks the floor.

| Dimension | Value |
|---|---|
| **Trigger** | Time-driven — daily cron at 06:00 (plant-local; configurable) |
| **Pattern** | Read → score → rank → notify (no writes to operational systems) |
| **Backend change required** | **None** — uses endpoints that already exist |
| **Blast radius** | Minimal — read-only; worst case is a missed/duplicate Slack post |
| **Outcome** | Supervisors start the shift with a ranked watch-list instead of a blank board |
| **KPI moved** | Time-to-first-intervention on at-risk orders; fewer "surprise" late orders |

This is the lowest-risk, highest-legibility workflow in the catalog, which is why
[doc 16](16_automation_architecture.md) sequences it **first**: it earns trust in the
automation layer before any workflow is allowed to *write* anything.

---

## 2. Workflow diagram

```
                          ┌─────────────────────┐
                          │  06:00 Daily Sweep  │  Schedule Trigger (cron 0 6 * * *)
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  GET /orders/today  │  HTTP · fullResponse · retry×3
                          └──────────┬──────────┘
                                     │ { body: OrderResponse[] }
                          ┌──────────▼──────────┐
                          │    Orders today?    │  IF  body.length >= 1
                          └─────┬─────────┬─────┘
                       true     │         │   false (no orders scheduled)
                                │         └──────────────┐
                   ┌────────────▼──────────┐    ┌─────────▼─────────┐
                   │     Split orders      │    │ Build empty digest│  "all clear, nothing to score"
                   └────────────┬──────────┘    └─────────┬─────────┘
                                │ 1 item / order          │
                   ┌────────────▼──────────┐               │
                   │ POST /{id}/evaluate   │  HTTP · batch 5@1s · retry · onError=continue
                   └────────────┬──────────┘               │
                                │ DelayPrediction          │
                   ┌────────────▼──────────┐               │
                   │  Enrich scored order  │  Set · join order meta + prediction
                   └────────────┬──────────┘               │
                   ┌────────────▼──────────┐               │
                   │   Sort by risk desc   │  Sort · delay_probability ↓
                   └────────────┬──────────┘               │
                   ┌────────────▼──────────┐               │
                   │ Roll up scored orders │  Aggregate · → single { orders:[…] } item
                   └────────────┬──────────┘               │
                   ┌────────────▼──────────┐               │
                   │   Build risk digest   │  Code · threshold filter + format + audit
                   └────────────┬──────────┘               │
                                └───────────┬──────────────┘
                                 ┌──────────▼──────────┐
                                 │  Slack configured?  │  IF  $env.SLACK_WEBHOOK_URL not empty
                                 └─────┬─────────┬──────┘
                              true     │         │   false
                       ┌──────────────▼──────┐   │
                       │ Notify: Slack digest │   │  HTTP · post text · onError=continue
                       └──────────────┬───────┘   │
                                      └─────┬─────┘
                                 ┌──────────▼──────────┐
                                 │      Audit log      │  NoOp · audit record persisted by n8n
                                 └─────────────────────┘
```

Both branches (orders found / no orders) converge on a **single notify + audit tail**, so
*every* run — busy day, quiet day, or all-clear — produces exactly one digest and one
audit record.

---

## 3. Node-by-node explanation

### Trigger

**`06:00 Daily Sweep`** — `scheduleTrigger`. Fires once per day on cron `0 6 * * *`.
The workflow `settings.timezone` is `UTC`; set it to the plant's timezone (e.g.
`Asia/Jakarta`) so 06:00 means 06:00 *on the floor*, not 06:00 UTC. A cron trigger (not an
interval) makes the run time explicit and auditable.

### HTTP Requests

**`GET /orders/today`** — `httpRequest`. Pulls the day's order list from the backend at
`http://backend:8000` (the Docker Compose service name — n8n and the API share the Compose
network, so no host ports or `localhost` are involved).

Two deliberate options:
- **`fullResponse: true`** wraps the reply as `{ statusCode, headers, body }` and returns it
  as **one** item. This is the key to empty-order handling: if the array were returned
  raw, an empty list would yield **zero** items and the whole chain would silently die.
  With `fullResponse`, we always get one item and can branch on `body.length`.
- **`retryOnFail` ×3 with a 3 s backoff** absorbs a transient backend hiccup at start-of-day.
  This node is intentionally *not* `continueOnFail`: a genuine backend outage should fail the
  run loudly (visible in n8n's execution log) rather than masquerade as "no orders".

**`POST /orders/{id}/evaluate`** — `httpRequest`. The scoring fan-out: one call per order,
URL templated with the order UUID. The `evaluate` endpoint assembles the full 38-feature
vector from the database itself, so n8n only needs the order ID — no feature engineering
leaks into the automation layer. Resilience options:
- **Batching — 5 items per batch, 1000 ms apart** (throttle). This is the model-saturation
  guard: instead of firing N concurrent scoring requests, n8n meters them so the ML service
  and DB stay healthy even on a heavy order day.
- **`retryOnFail` ×3** for transient errors, and **`onError: continueRegularOutput`** so a
  single un-scoreable order does **not** abort the sweep. That order flows on with
  `scored_ok = false` and is reported as a skipped/failed score in the digest.

**`Notify: Slack digest`** — `httpRequest`. Posts the composed digest text to the Slack
Incoming Webhook at `$env.SLACK_WEBHOOK_URL`. `onError: continueRegularOutput` means a Slack
outage never fails the sweep — the audit record is still written.

### Filters

The workflow filters in **two** places, by design:

1. **`Orders today?`** — `IF` node. The empty-vs-non-empty gate (`body.length >= 1`). The
   `false` branch is the empty-order path.
2. **`Slack configured?`** — `IF` node. Routes to the Slack post only when
   `$env.SLACK_WEBHOOK_URL` is set; otherwise it skips straight to audit. This lets the
   workflow be imported and run end-to-end *before* Slack is wired up.
3. **Risk threshold** (`delay_probability >= 0.70`) — applied inside `Build risk digest`
   rather than as a standalone Filter node. **Why in code:** a native `Filter` node emits
   zero items when nothing matches, which would silently skip the notify + audit tail on a
   zero-high-risk morning. Filtering on the aggregated single item guarantees an "all clear"
   digest and an audit record are still produced. (`0.70` mirrors the backend
   `HIGH_RISK_THRESHOLD`.)

### Sort

**`Sort by risk desc`** — `sort`. Orders the scored items by `delay_probability` descending,
so the riskiest order is first. The order is preserved through aggregation, so the digest
reads worst-first without any re-sorting on the consumer side.

### Aggregation

**`Roll up scored orders`** — `aggregate` in *Aggregate All Item Data* mode. Collapses the N
per-order items into **one** item shaped `{ orders: [ …all scored orders… ] }`. This
single-item convergence is what makes the downstream digest deterministic and lets one Code
node compute every count (scanned, high-risk, failed) in one place.

Two supporting transforms feed it:
- **`Enrich scored order`** — `set`. Joins each prediction back to its order context. The
  `evaluate` response carries only the prediction, so this node pulls `order_number`,
  `priority`, and `planned_start` from the paired upstream item
  (`$('Split orders').item`) and flattens them alongside `delay_probability`, `root_cause`,
  the top SHAP `top_factor`, and a `scored_ok` flag.
- **`Split orders`** — `splitOut`. Expands the `body` array from the GET into one item per
  order so the scoring fan-out can run per-order.

### Notifications

**`Build risk digest`** — `code`. Reads the aggregated `orders`, selects high-risk
(`scored_ok && delay_probability >= 0.70`), formats the Slack message (worst-first, with
percentage, root cause, priority, and the dominant SHAP driver per order), and assembles the
**audit** object. On a clean morning it emits a `:white_check_mark:` all-clear; on a risky
one a `:rotating_light:` ranked list. Failed scores are surfaced as a footnote, never hidden.

**`Build empty digest`** — `set`. The no-orders counterpart. Produces the same
`{ slack_text, audit }` shape so the notify + audit tail is shared, not duplicated.

**`Audit log`** — `noOp`. The convergence point. Its input item carries the `audit` record,
which n8n persists in the execution log (`settings.saveDataSuccessExecution: "all"`). Every
run is therefore queryable after the fact — what was scanned, how many were high-risk, the
threshold used, and the run ID — without any backend table change. See §7 for the optional
upgrade to backend-persisted audit rows.

---

## 4. Setup

### 4.1 Prerequisites

- The backend stack running under Docker Compose, with n8n on the **same Compose network**
  (so `http://backend:8000` resolves). This matches the existing
  [`high-risk-order-alert.json`](../n8n/workflows/high-risk-order-alert.json) workflow.
- A Slack **Incoming Webhook** URL (optional — the workflow runs without it and simply skips
  the post).

### 4.2 Environment variable

Expose the Slack webhook to the n8n container as an environment variable (the workflow reads
`$env.SLACK_WEBHOOK_URL`, never a hard-coded URL):

```yaml
# docker-compose.yml — n8n service
services:
  n8n:
    environment:
      - SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL}   # same value the backend uses
      - GENERIC_TIMEZONE=Asia/Jakarta            # so 06:00 cron = plant-local 06:00
```

> n8n only exposes environment variables to expressions when they are allowed. If
> `$env.SLACK_WEBHOOK_URL` reads empty inside n8n, ensure the variable is **not** blocked
> (i.e. `N8N_BLOCK_ENV_ACCESS_IN_NODE` is unset/`false`) and restart the container.

### 4.3 Import

1. n8n → **Workflows** → **Import from File**.
2. Select [`n8n/workflows/morning-risk-sweep.json`](../n8n/workflows/morning-risk-sweep.json).
3. Confirm `settings → timezone` (default `UTC`) matches your plant; adjust the cron if you
   want a time other than 06:00.
4. **Activate** the workflow to arm the daily schedule.

No credentials object is required — both the backend and Slack calls are plain HTTP to URLs
the workflow already knows.

---

## 5. Testing procedure

### 5.1 Dry run (manual, no schedule)

1. Open the workflow and click **Execute Workflow** (the manual trigger bypasses the cron).
2. Watch the data pills accumulate down the canvas. Inspect each node's output:
   - `GET /orders/today` → one item with `body` = the order array and `statusCode: 200`.
   - `POST /orders/{id}/evaluate` → one item per order, each a `DelayPrediction`.
   - `Build risk digest` → one item with `slack_text` + `audit`.
3. Open `Audit log` and confirm the `audit` object (counts, threshold, run ID, high-risk
   list) is present.

### 5.2 Scenario coverage

| Scenario | How to produce it | Expected behaviour |
|---|---|---|
| **High-risk orders present** | Normal day with at-risk orders | `:rotating_light:` digest, ranked worst-first; audit `high_risk_count > 0` |
| **Orders present, none high-risk** | A day where all scores < 0.70 | `:white_check_mark:` all-clear digest still sent; audit `high_risk_count: 0` |
| **No orders scheduled** | Empty `GET /orders/today` | `Build empty digest` path; "nothing to score" message; audit `note: no_orders_scheduled` |
| **One order fails to score** | Point one order at a missing FK / force a 500 | Sweep completes; that order `scored_ok=false`; digest footnote "⚠ 1 order(s) could not be scored" |
| **Slack not configured** | Unset `SLACK_WEBHOOK_URL` | `Slack configured?` routes to audit; run still succeeds, no post |
| **Backend down** | Stop the `backend` container | `GET` retries 3× then the **run fails loudly** in the execution log (correct — this is an outage, not "no orders") |

### 5.3 Schedule verification

After activating, confirm the next scheduled execution appears in **Executions**, and let
one fire (or temporarily set the cron a minute ahead). A green execution with a populated
`Audit log` output confirms end-to-end health.

---

## 6. Expected outputs

### 6.1 Slack digest — high-risk morning

```
:rotating_light: Morning Risk Sweep — 3 high-risk of 11 order(s) scheduled today (threshold ≥ 70%)
1. WO-20846 — 91% delay risk | machine_overload | CRITICAL | driver: Work-center queue depth at release
2. WO-20851 — 78% delay risk | material_shortage | HIGH | driver: Component shortage count
3. WO-20833 — 72% delay risk | operator_load | NORMAL | driver: Operator concurrent order count
```

### 6.2 Slack digest — all clear

```
:white_check_mark: Morning Risk Sweep — 0 high-risk of 9 order(s) scheduled today. All clear.
```

### 6.3 Slack digest — no orders

```
:white_check_mark: Morning Risk Sweep (2026-06-14) — No production orders scheduled today. Nothing to score.
```

### 6.4 Audit record (persisted in the n8n execution log)

```json
{
  "workflow": "morning_risk_sweep",
  "run_id": "1284",
  "trigger": "schedule",
  "scanned_at": "2026-06-14T06:00:04.812Z",
  "orders_scanned": 11,
  "high_risk_count": 3,
  "scoring_failed_count": 0,
  "threshold": 0.7,
  "high_risk_orders": [
    { "order_id": "…", "order_number": "WO-20846", "delay_probability": 0.91, "root_cause": "machine_overload", "priority": "critical" },
    { "order_id": "…", "order_number": "WO-20851", "delay_probability": 0.78, "root_cause": "material_shortage", "priority": "high" },
    { "order_id": "…", "order_number": "WO-20833", "delay_probability": 0.72, "root_cause": "operator_load", "priority": "normal" }
  ]
}
```

---

## 7. Design notes & next steps

- **Why no native Filter node for the threshold:** see §3 → Filters. Robustness (always emit
  a digest + audit) beats node purity here.
- **Audit persistence:** today the audit lives in n8n's execution log — durable, queryable,
  zero backend change, true to the W2 "no code" constraint. To surface sweeps in the
  backend's own `WorkflowExecution` table (and the existing
  `GET /api/v1/webhooks/executions` view), add the thin `ops/audit` endpoint planned in
  [doc 16 §6](16_automation_architecture.md) and append one HTTP node after `Audit log`.
  That is a **Phase 2** enhancement, deliberately out of scope for W2.
- **Throughput headroom:** batching at 5@1s comfortably covers a typical daily order count.
  For very large plants, raise `batchSize` or switch the fan-out to the batch prediction
  endpoint — but that requires assembling `OrderFeatures` in n8n, which is exactly the
  coupling `evaluate` was built to avoid.

This workflow establishes the reusable **Trigger → Fetch → Decide → Act → Audit** skeleton
that W3 (Shift-End Digest) and W5 (Bottleneck Escalation) extend.

## W2 Sign-Off Status

Status: Complete with Deferred Slack Integration

Date: 2026-06-14

Reason:
No production Slack webhook has been provisioned.

Current Behavior:

* Workflow imports successfully.
* Risk scoring executes successfully.
* Digest generation executes successfully.
* Audit path executes successfully.
* Slack notification path is deferred until webhook provisioning.

Known Gap:
SLACK_WEBHOOK_URL is not configured in the deployment environment.

Owner:
Infrastructure / Deployment Configuration

Future Action:
Provision Slack webhook and complete notification validation test (T3).

---

*Document 17 — Manufacturing Process Copilot Technical Series*
