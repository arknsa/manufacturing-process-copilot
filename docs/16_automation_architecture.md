# Document 16 — Business Process Automation Architecture

**Status:** Active
**Purpose:** define how the Manufacturing Process Copilot evolves from an AI/ML
*showcase* into a **manufacturing automation platform** — by wrapping the existing
prediction, analytics, and recommendation capabilities in an event-driven n8n
orchestration layer. This document is architecture and planning only; it changes
no code.

**Scope:** generic manufacturing operations (order intake, scheduling,
maintenance, bottleneck management, shift reporting). No company-specific logic.
No RAG. No additional ML models. Automation and business-process orchestration
only.

---

## 1. Automation Vision

### 1.1 The shift in framing

V1 answers a question: *"Will this order be late, and why?"* That is **detection**.
A detection system that requires a human to read a dashboard, interpret it, and
act is only half a system. The business value of manufacturing software is
realized when the loop closes without a person in the critical path:

> **detect → decide → act → escalate → verify → report**

The Copilot already does *detect* (delay prediction + SHAP root cause) and
*decide-support* (the ReAct agent). The automation layer adds **act, escalate,
verify, and report** as deterministic, auditable, scheduled processes.

### 1.2 Design principles

1. **Deterministic orchestration.** Scheduled and event-driven flows must be
   reproducible. The LLM agent is for *human conversation*, not for cron jobs.
   Workflows call typed REST endpoints, not the chat agent, for operational data.
2. **The backend stays thin; n8n owns choreography.** Report formatting,
   branching, retries, and notification fan-out live in n8n. The backend exposes
   capabilities, not workflows.
3. **Every workflow is audited.** Each run writes a `WorkflowExecution` row
   (table already exists) so automation is observable, not magic.
4. **Idempotent and storm-safe.** Scheduled sweeps must dedup so a recurring
   trigger cannot spam the same alert every 30 minutes.
5. **Graceful degradation.** A missing Slack URL, an empty order list, or a model
   timeout must downgrade to a no-op or a logged failure — never a crash.

### 1.3 Architectural finding that shapes everything

The operational analytics — **bottleneck detection, shift summary, KPI dashboard,
recommendation lifecycle** — are **not REST endpoints today**. They live in the
agent tool layer (`backend/app/services/agent/tools/analytics.py` and
`recommendations.py`) and are reachable **only through the LLM**. Orchestration
needs deterministic access, so the **single highest-leverage enabler** is a thin
read/write `ops` router that wraps those existing functions as plain REST. This is
the one structural dependency for the reporting and escalation workflows and is
called out explicitly in the roadmap (Section 8).

---

## 2. Workflow Catalog

Ten workflows span the full manufacturing automation surface: intake, planning,
maintenance, bottleneck management, procurement, SLA enforcement, approvals,
quality feedback, and management reporting.

| # | Workflow | Trigger | Core APIs | Automates |
|---|---|---|---|---|
| **W1** | New-Order Intake Triage & Auto-Routing | Webhook on order create | `/orders/{id}/evaluate` → branch on probability + root_cause → `ops/recommendations`, status `PATCH` | Release gating: auto-approve low-risk, hold + flag high-risk |
| **W2** | Morning Schedule Risk Sweep | Cron 05:30 daily | `/orders/today` → loop `/orders/{id}/evaluate` → rank | Prioritized daily risk watch-list to the planner |
| **W3** | Shift-End Digest & Handover Report | Cron 05:55 / 13:55 / 21:55 | `ops/shift-summary`, `ops/kpi`, `ops/recommendations` → Slack / email / Drive | Real shift handover document (replaces the stubbed webhook) |
| **W4** | Predictive Maintenance Work-Order Trigger | Cron hourly | `ops/machine-history` thresholds (OEE / downtime / days-since-PM) → ticket + notify | Threshold-based PM work-order creation |
| **W5** | Active Bottleneck Escalation & Ticketing | Cron every 30 min | `ops/bottlenecks` → `ops/recommendations` + ticket + tiered notify | Closed loop: detect → recommend → ticket → notify by severity |
| **W6** | Material Shortage → Procurement Escalation | Branch off W1 / cron | `evaluate` → `root_cause == material/shortage` → PO request + procurement notify | Routes shortage risk to purchasing |
| **W7** | Expedited / SLA Order Watchdog | Cron every 15 min | `/orders/today` filtered → re-score → escalation ladder | Time-boxed escalation until acknowledged |
| **W8** | Recommendation Approval (Human-in-the-Loop) | Webhook on high-urgency rec | Slack interactive / email → callback webhook → `ops/recommendations` `PATCH` | Approval routing + decision capture |
| **W9** | Nightly Outcome Feedback & Accuracy Monitor | Cron 23:00 | join predictions vs `status` → rolling accuracy → `/webhooks/feedback-loop` + alert | Drift / quality monitoring (no training) |
| **W10** | Weekly Management KPI Report | Cron Mon 07:00 | `ops/kpi`, `webhooks/executions`, bottleneck trend → HTML/PDF → email + Drive | Executive reporting pack |

**Baseline already shipped:** *High-Risk Order Alert*
(`n8n/workflows/high-risk-order-alert.json`) — the reactive single-order Slack
alert. The catalog above extends that pattern from *one reactive alert* to a
*scheduled, closed-loop operations layer*.

---

## 3. Workflow Prioritization

Scored 1–5 (5 = best). **Priority = Business Value + Interview Impact − Effort**,
weighted toward workflows that complete already-half-built functionality.

| # | Workflow | Business Value | Interview Impact | Effort (lower = better) | Backend change | Verdict |
|---|---|:--:|:--:|:--:|:--:|---|
| **W2** | Morning Risk Sweep | 5 | 5 | 2 | **None** | **Build 1st** |
| **W3** | Shift-End Digest | 5 | 4 | 2 | thin `ops` reads | **Build 2nd** |
| **W5** | Bottleneck Escalation | 5 | 5 | 3 | thin `ops` reads | **Build 3rd** |
| W1 | Intake Triage | 4 | 4 | 3 | small | Strong next |
| W4 | Predictive Maintenance WO | 5 | 4 | 3 | thin `ops` read | Strong next |
| W8 | Recommendation Approval | 4 | 5 | 4 | callback wiring | High wow, more effort |
| W7 | SLA Watchdog | 4 | 3 | 3 | none | Solid |
| W9 | Feedback / Accuracy Monitor | 4 | 4 | 4 | feedback logic | MLOps story |
| W6 | Material → Procurement | 3 | 3 | 2 | none | Quick add-on to W1 |
| W10 | Weekly Management Report | 3 | 3 | 3 | none | Nice-to-have |

**Reading the table:** the top cluster (W2/W3/W5) maximizes value and interview
impact while sharing a single enabler (the `ops` router), so the marginal cost
falls after the first build. The mid-tier (W1/W4/W8) are the natural Phase 2.
W6/W10 are low-effort polish that can be slotted in opportunistically.

---

## 4. Event-Driven Architecture

### 4.1 Two trigger classes

The automation layer is built on exactly two event sources, which keeps the mental
model simple:

- **Time-driven (cron):** planning and reporting cadences — morning sweep, shift
  boundaries, hourly maintenance checks, nightly feedback, weekly report.
- **State-change (webhook):** an order is created, a recommendation crosses an
  urgency threshold, a human approves/rejects.

```
                          ┌──────────────────────────────────────────┐
   TIME (cron)            │                  n8n                      │
   05:30 ─ Morning Sweep ─┤  Trigger → Fetch → Branch → Act → Notify  │
   xx:55 ─ Shift End ─────┤            → Audit                        │
   :/30 ─ Bottleneck ─────┤                                          │
                          └───────────────┬──────────────────────────┘
                                          │ HTTP (deterministic REST)
   STATE (webhook)                        ▼
   order.created ─────────────► ┌───────────────────────────┐
   rec.high_urgency ──────────► │   FastAPI backend          │
   human.decision ────────────► │   /orders  /predictions    │
                                │   /ops (new)  /webhooks     │
                                └─────────────┬──────────────┘
                                              │
                                  ┌───────────┴───────────┐
                                  ▼                       ▼
                            PostgreSQL              ML service
                       (orders, machines,        (delay model +
                        bottlenecks, recs,         SHAP root cause)
                        workflow_executions)
```

### 4.2 The canonical workflow shape

Every workflow in the catalog reduces to the same five-stage skeleton, which makes
them easy to build, review, and explain:

```
TRIGGER → FETCH (REST) → DECIDE (filter/switch) → ACT (notify/ticket/PATCH) → AUDIT (WorkflowExecution)
```

- **TRIGGER** — cron or webhook.
- **FETCH** — read current state via deterministic REST (`/orders/today`,
  `/ops/bottlenecks`, …). Never the chat agent.
- **DECIDE** — n8n `if` / `switch` / `filter` on thresholds
  (`PREDICTION_THRESHOLD = 0.65`, `HIGH_RISK_THRESHOLD = 0.70`, severity tiers).
- **ACT** — Slack/email fan-out, external ticket/PO creation, recommendation
  create, order status `PATCH`.
- **AUDIT** — POST to a webhook that writes a `WorkflowExecution` row.

### 4.3 Idempotency & safety

- **Dedup state:** scheduled sweeps (W5, W7) record processed entity IDs (n8n
  static data or a Postgres marker) so a 30-minute cron does not re-escalate the
  same bottleneck every cycle.
- **Batching:** the morning sweep (W2) batches `evaluate` calls (e.g. 5 at a time)
  to avoid saturating the model service.
- **Empty-set handling:** an empty `/orders/today` or zero active bottlenecks
  resolves to a clean no-op (optionally an "all clear" message), never an error.

---

## 5. n8n Integration Layer

### 5.1 Network & conventions

- **Internal host:** workflows call `http://backend:8000` (Docker Compose service
  name), matching the existing `high-risk-order-alert.json`.
- **Node standard:** `n8n-nodes-base.*`, `executionOrder: v1`, expression syntax
  `={{ ... }}`.
- **Repo layout:** exported workflow JSON lives in `n8n/workflows/`, version-
  controlled alongside the baseline alert workflow.

### 5.2 Responsibility split

| Concern | Owner | Rationale |
|---|---|---|
| Scheduling / cron cadence | n8n | Operational, not application logic |
| Branching & thresholds | n8n | Tunable without redeploying the backend |
| Notification fan-out (Slack/email/Drive) | n8n | Channel routing is ops config |
| External systems (CMMS/ERP/Sheets/Trello) | n8n | Integration glue, kept out of core |
| Capabilities (predict, score, analytics, recs) | FastAPI | Business logic + data access |
| Persistence & audit | PostgreSQL | Single source of truth |
| Human-readable narrative | LLM agent | Only where conversation/explanation is needed |

### 5.3 Secrets & configuration

Driven by existing settings in `backend/app/core/config.py`:
`SLACK_WEBHOOK_URL`, `ALERT_EMAIL_TO`, `HIGH_RISK_THRESHOLD`,
`PREDICTION_THRESHOLD`. External-system credentials (CMMS, Google, Trello/Jira)
are held as **n8n credentials**, never in the backend.

---

## 6. API Dependencies

### 6.1 Available today (workflows can call directly)

| Endpoint | Used by |
|---|---|
| `POST /api/v1/orders/{id}/evaluate` — assembles features from DB + scores by UUID | W1, W2, W7 |
| `GET /api/v1/orders/today` | W2, W7 |
| `PATCH /api/v1/orders/{id}/status` | W1 |
| `POST /api/v1/predictions/delay[/batch]` | W2 (alt.), what-if |
| `POST /api/v1/webhooks/order-released` | W1 (baseline alert path) |
| `POST /api/v1/webhooks/shift-end` | W3 (audit sink) |
| `POST /api/v1/webhooks/feedback-loop` | W9 (audit sink) |
| `GET /api/v1/webhooks/executions` | W10 (observability) |
| `GET /api/v1/models/current`, `/feature-importance` | W9, W10 |

### 6.2 Required enabler — thin `ops` router (no new business logic)

Wraps functions that **already exist** in the agent tool layer:

| New endpoint | Wraps (existing function) | Unblocks |
|---|---|---|
| `GET /api/v1/ops/bottlenecks?active_only=` | `analytics.get_bottlenecks` | W5 |
| `GET /api/v1/ops/shift-summary?shift_date=` | `analytics.get_shift_summary` | W3 |
| `GET /api/v1/ops/kpi` | `analytics.get_kpi_dashboard` | W3, W10 |
| `GET /api/v1/ops/machine-history?machine_code=` | `analytics.get_machine_history` | W4 |
| `GET/POST/PATCH /api/v1/ops/recommendations` | `recommendations.*` | W3, W5, W8 |

> These functions take `(db, ...args)` and return plain dicts today; the router is
> dependency-injection plumbing, not new logic. This is the one architectural
> prerequisite for the reporting/escalation tier.

### 6.3 Dependency map (top-3)

```
W2 Morning Sweep   ──► /orders/today, /orders/{id}/evaluate        [no backend change]
W3 Shift Digest    ──► /ops/shift-summary, /ops/kpi, /ops/recs     [needs ops router]
                       /webhooks/shift-end (audit)
W5 Bottleneck Esc. ──► /ops/bottlenecks, /ops/recommendations      [needs ops router]
```

---

## 7. Business Outcomes

| Workflow | Operational outcome | KPI moved |
|---|---|---|
| W2 Morning Sweep | Planner starts the day with a ranked risk list instead of a blank dashboard | Earlier intervention → on-time delivery rate |
| W3 Shift Digest | Every shift handover is consistent, complete, and archived | Handover quality; fewer dropped issues across shifts |
| W5 Bottleneck Escalation | Constraints become tickets with owners automatically | Mean-time-to-acknowledge; bottleneck dwell time |
| W1 Intake Triage | Risky orders are held/flagged at release, not discovered late | Reduced expedite cost, fewer schedule revisions |
| W4 Predictive Maintenance | PM work orders fire on condition thresholds | Unplanned downtime hours |
| W7 SLA Watchdog | Expedited/critical orders cannot silently slip | Expedited-order on-time rate |
| W8 Approval HITL | Recommendations get a decision and an audit trail | Recommendation action rate |
| W9 Feedback Monitor | Model accuracy is tracked against real outcomes | Sustained prediction quality / drift detection |
| W10 Weekly Report | Management gets a standing performance pack | Visibility; decision latency |

**Portfolio framing:** the system stops being "an ML model with a dashboard" and
becomes "an event-driven operations platform where predictions trigger
deterministic, audited business processes." That is the difference between a data
science demo and a manufacturing automation product.

---

## 8. Top-3 Comparison & First-Build Decision

### 8.1 Side-by-side

| Dimension | **W2 — Morning Risk Sweep** | **W3 — Shift-End Digest** | **W5 — Bottleneck Escalation** |
|---|---|---|---|
| Trigger | Cron 05:30 daily | Cron 3×/day (shift boundaries) | Cron every 30 min |
| Primary value | Proactive daily planning | Consistent shift handover | Closed-loop constraint resolution |
| Pattern demonstrated | Fan-out scoring + ranking | Parallel fetch + aggregate + multi-channel | Detect → recommend → ticket → tiered notify |
| Backend change | **None** | Thin `ops` reads | Thin `ops` reads |
| External systems | Slack + Sheet | Slack + email + Drive | Slack + CMMS/ticketing |
| Statefulness | Stateless | Stateless | **Stateful** (dedup required) |
| Failure blast radius | Low (read + notify) | Low (read + notify) | Medium (creates tickets/recs) |
| Implementation effort | **Lowest** | Low | Medium |
| Interview impact | High ("the factory plans its day") | High ("completes a half-built feature") | **Highest** ("full BPA loop") |

### 8.2 Trade-off read

- **W5** has the strongest narrative (a genuine closed loop) but carries the most
  risk and complexity: it *writes* (recommendations, external tickets) and
  **requires dedup state** to avoid alert storms. Building it first means debugging
  orchestration, the `ops` router, and idempotency simultaneously.
- **W3** is high value and completes the already-stubbed `shift-end` webhook, but
  it depends on the `ops` router and touches three external channels (Slack, email,
  Drive) — more integration surface on day one.
- **W2** delivers the most visible win with the **lowest risk and zero backend
  change**. It is read-only plus notify, stateless, and exercises the canonical
  workflow skeleton end-to-end. It is the ideal first build to validate the
  pattern, the Docker networking, and the audit path before anything *writes*.

### 8.3 Recommendation — build **W2 (Morning Risk Sweep) first**

**Rationale:**

1. **Zero backend dependency** — ships against endpoints that exist today, so it
   decouples "prove the automation layer works" from "add the `ops` router."
2. **Lowest blast radius** — read + notify only; nothing is written to external
   systems, so a bug cannot create spurious tickets or corrupt state.
3. **Establishes the reusable skeleton** — TRIGGER → FETCH → DECIDE → ACT → AUDIT
   is built and proven once, then copied for W3 and W5.
4. **Highest immediate, demonstrable value** — a ranked morning risk list is the
   most legible "automation earns its keep" moment for a stakeholder or
   interviewer.

**Sequenced conclusion:** build **W2** to validate the pattern, then land the
**`ops` router** as the shared enabler, then **W3**, then **W5** (with dedup).
This orders the work from lowest-risk/no-dependency to highest-complexity/stateful.

---

## 9. Implementation Roadmap

Four phases. Each phase is independently demoable and leaves the system in a
shippable state. No code is written in this document — this is the plan.

### Phase 0 — Foundation & conventions (0.5 day)

- Confirm Docker Compose networking: n8n → `http://backend:8000` reachable
  (baseline alert workflow already proves this).
- Define the **audit contract**: every workflow ends with a POST that writes a
  `WorkflowExecution` row (`workflow_name`, `trigger_type`, `status`,
  `input_data`, `output_data`).
- Establish the `n8n/workflows/` export discipline (commit JSON per workflow).
- **Exit criteria:** a one-node test workflow writes an audit row and appears in
  `GET /api/v1/webhooks/executions`.

### Phase 1 — W2 Morning Risk Sweep (1 day) — *no backend change*

- Build the canonical skeleton: Schedule Trigger (05:30) → `GET /orders/today` →
  Split → batched `POST /orders/{id}/evaluate` → merge order+score → filter
  `≥ 0.65` → sort desc → build watch-list → Slack + Sheet → audit.
- Add empty-set "all clear" branch.
- **Exit criteria:** on a seeded day, the planner channel receives a ranked list;
  a trend row lands in the Sheet; an audit row is written. Re-running is safe.

### Phase 2 — `ops` router enabler (0.5–1 day) — *backend, no new logic*

- Add `backend/app/api/routes/ops.py` exposing read/write wrappers over the
  existing `analytics.*` and `recommendations.*` functions; register under
  `/api/v1` in `main.py`.
- Endpoints: `/ops/bottlenecks`, `/ops/shift-summary`, `/ops/kpi`,
  `/ops/machine-history`, `/ops/recommendations` (GET/POST/PATCH).
- **Exit criteria:** each endpoint returns the same payload the agent tool returns
  today, verified against a seeded database; OpenAPI docs render the new routes.

### Phase 3 — W3 Shift-End Digest (1 day)

- Schedule Trigger (3 crons) → resolve shift label/date → **parallel** fetch
  (`/ops/shift-summary`, `/ops/kpi`, `/ops/recommendations`) → merge → build digest
  → Slack + email + Drive archive → `POST /webhooks/shift-end` (audit).
- **Exit criteria:** a handover document is delivered to all three channels at a
  shift boundary and archived; the existing stub's audit row is populated with real
  digest data.

### Phase 4 — W5 Bottleneck Escalation (1.5 days)

- Schedule Trigger (every 30 min) → `GET /ops/bottlenecks?active_only=true` →
  Split → **dedup guard** → `switch` on severity → (critical/high) create
  recommendation + open ticket + tiered notify → (medium/low) log → mark escalated
  → audit.
- Implement dedup via Postgres marker or n8n static data; tune notification tiers.
- **Exit criteria:** a seeded critical bottleneck produces exactly one
  recommendation + one ticket + one tiered alert per detection; a second run within
  the window does **not** re-escalate.

### Phase 5 — Documentation & demo (0.5 day)

- Update `README.md` automation section and link this document.
- Record a walkthrough: cron fires → REST fetch → branch → notify → audit row.
- Capture before/after framing ("ML demo" → "event-driven operations platform").

### Timeline summary

| Phase | Deliverable | Effort | Backend change |
|---|---|---|---|
| 0 | Foundation + audit contract | 0.5 d | none |
| 1 | **W2 Morning Risk Sweep** | 1 d | none |
| 2 | `ops` router enabler | 0.5–1 d | wrappers only |
| 3 | W3 Shift-End Digest | 1 d | none (uses Phase 2) |
| 4 | W5 Bottleneck Escalation | 1.5 d | none (uses Phase 2) |
| 5 | Docs + demo | 0.5 d | none |
| | **Total** | **~5 days** | **1 thin router** |

### Risks & mitigations

| Risk | Mitigation |
|---|---|
| Alert storms from scheduled sweeps | Dedup state (Phase 4); batching (Phase 1) |
| Model service saturation from bulk scoring | Batch `evaluate` calls with interval throttling |
| External integration flakiness (Slack/Drive/CMMS) | Per-node timeouts + retry; degrade to audit-only |
| `ops` router drift from agent tools | Router wraps the *same* functions — single source of truth |
| Non-reproducible runs | Deterministic REST (no LLM in cron paths) + audit rows |

---

## 10. Summary

The Copilot already detects and explains delay risk. This document defines the
**automation layer** that turns those predictions into deterministic, audited
business processes — a ten-workflow catalog, an event-driven architecture built on
two trigger classes and one canonical skeleton, and a clean responsibility split
between n8n (choreography) and FastAPI (capability).

The recommended first build is **W2 — Morning Risk Sweep**: highest immediate
value, lowest risk, and zero backend change, establishing the reusable pattern.
The **`ops` router** then unlocks **W3** and **W5**, completing the move from an
AI showcase to a manufacturing automation platform in roughly one focused week.
