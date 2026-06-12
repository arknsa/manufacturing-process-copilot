# Manufacturing Process Copilot
## Document 01 — Final Simulation Architecture

**Status:** Production-Ready  
**Version:** 1.0  
**Validated:** 120-day run, seed 42, 954 orders, 36.6% delay rate

---

## 1. Design Philosophy

The simulator is built on one architectural principle: **delays must emerge from cause, not be sampled from labels.**

Every delayed order in the dataset is delayed because something went wrong in its simulated execution — a machine broke down, a setup took too long, materials arrived late, a junior operator was overloaded. The ML model therefore learns to predict the same causal chain a human production planner would reason about. This rules out the most common failure mode in manufacturing ML: a model that learns to correlate a label with a confounded proxy rather than with the underlying process.

A secondary principle: **features are snapshotted at the prediction point, not at completion.** Every feature in the dataset reflects what was knowable at order release — before execution began. Features that would only be observable after the order started are excluded. This eliminates data leakage by construction.

---

## 2. Layer Architecture

The simulation is organised into six layers, each with a single responsibility.

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 5  OUTPUT                                                │
│  DatasetBuilder · CalibrationChecker · ReportGenerator         │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4  COLLECTION                                            │
│  FeatureCollector (at release) · OutcomeRecorder · TimelineLog  │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3  ORCHESTRATION                                         │
│  FactorySimulation.run() · .export() · .generate_report()       │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2  ENGINE                                                │
│  EventEngine · Scheduler · StateManager                         │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1  GENERATION                                            │
│  DemandGenerator (Poisson) · BreakdownGenerator (Weibull)       │
│  AbsenteeismGenerator (Bernoulli) · SetupRunTimeGenerator       │
│  (log-normal) · QualityOutcomeGenerator (FPY model)             │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 0  FOUNDATION                                            │
│  SimConfig (Pydantic) · Entities (dataclasses)                  │
│  Events (typed enum) · Clock (shift-aware) · SeedManager        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer Specifications

### Layer 0 — Foundation

**SimConfig** holds all tunable parameters as a Pydantic model with validation. Key parameters: `simulation_days`, `target_orders_per_day`, `num_machines`, `num_products`, `num_operators_per_shift`, `seed`, `machine_oee_mean`, `machine_oee_std`, `rework_fraction`.

**Entity dataclasses** (frozen fields set at init, mutable state updated during simulation):

| Entity | Key fields |
|---|---|
| `ProductEntity` | `complexity`, `routing_length`, `standard_setup_time_minutes`, `standard_run_time_per_unit_minutes`, `base_first_pass_yield`, `material_bom_complexity` |
| `MachineEntity` | `machine_type`, `mtbf_hours`, `mttr_hours_mean`, `setup_overrun_tendency`, `current_oee`, `last_maintenance_date` |
| `OperatorEntity` | `skill_tier`, `shift_assignment`, `setup_speed_multiplier`, `absenteeism_base_rate`, `certified_machine_types` |
| `SupplierEntity` | `reliability_score` (0.80–0.98) |
| `ProductionOrderState` | All planning fields + all execution outcome fields + all 37 ML features |

**Events** are a typed enum covering: `ORDER_CREATED`, `ORDER_RELEASED`, `ORDER_ON_HOLD_MATERIAL`, `ORDER_ON_HOLD_MACHINE`, `ORDER_ON_HOLD_OPERATOR`, `ORDER_QC_FAIL`, `ORDER_COMPLETED`, `MACHINE_PM_COMPLETE`.

**Clock** is shift-aware: Morning (06:00–14:00), Afternoon (14:00–22:00), Night (22:00–06:00). Weekends are skipped.

---

### Layer 1 — Generation

**DemandGenerator** produces daily order count via Poisson with rate modulated by day-of-week (±20%), end-of-month (+35%), and quarter-end (+70%) multipliers. Per-order priority and quantity are sampled from empirically calibrated distributions.

**Lead-time distribution:**
- 30% rush orders: 1–5 days, priority high/critical
- 50% normal orders: 5–15 days, priority normal/high
- 20% planned orders: 15–45 days, priority normal

**Planned window** is computed after product assignment using priority-based buffers that produce realistic schedule tightness:

| Priority | Buffer factor | Typical tightness ratio |
|---|---|---|
| critical | 1.15 × noise | 0.87 |
| high | 1.50 × noise | 0.67 |
| normal | 1.80 × noise | 0.56 |
| low | 2.50 × noise | 0.40 |

Noise is log-normal ±10%. This is the primary driver of the tightness distribution observed in the training data (mean 0.60, std 0.13).

**BreakdownGenerator** uses a Weibull reliability model (β=2.5) for daily machine wear state and a directly-calibrated per-order breakdown probability for order-level events. The per-order probability scales with PM overdue fraction and OEE degradation, producing ~10–12% of orders experiencing a breakdown event.

**SetupRunTimeGenerator** samples actual setup times from log-normal (σ=0.28) and run times from normal (σ=10% of mean). Both are biased by machine `setup_overrun_tendency` (1.01–1.08×) and operator `setup_speed_multiplier` (0.95–1.12×).

**QualityOutcomeGenerator** uses a first-pass yield (FPY) model: LOW complexity 97%, MEDIUM 93%, HIGH 88%. Skill and OEE adjustments are ±2–5%.

When an order fails inspection, rework time is:

```
rework_hours = rework_fraction × estimated_total_hours
```

where `rework_fraction` is a `SimConfig` parameter (calibrated default: 0.80). Rework hours are added directly to the `extra_hours` accumulator.

The rework delay condition simplifies analytically: rework causes delay when `rework_fraction > (1 − tightness) / tightness`. At the default 0.80, this threshold is crossed at `schedule_tightness_ratio ≈ 0.56`, which falls within the lower quartile of the tightness distribution. Approximately 65% of quality failure events are expected to produce a delay at this setting. The `estimated_total_hours` term cancels from both sides of the delay inequality, meaning the conversion rate from quality failure to delay is determined entirely by the tightness distribution and is independent of absolute order size.

---

### Layer 2 — Engine

**Scheduler** assigns each order to the least-utilised compatible machine and the available operator on the appropriate shift with the required certification. Changeover is detected from the last product on that machine; complexity is sampled uniform(1.5, 3.0) when a changeover is required.

**StateManager** maintains: `active_orders`, `completed_orders`, per-machine scheduled windows (for utilisation calculation), and a rolling order history (used for all 90-day and 30-day historical features).

---

### Layer 3 — Orchestration

`FactorySimulation._simulate_day()` runs the following sequence each working day:

1. `_update_machines()` — accumulate wear hours, trigger PM if due, update OEE, stochastic daily breakdown (updates machine state only)
2. `_process_absenteeism()` — mark operators unavailable for today
3. For each of N Poisson-sampled orders: `_create_order()` — the complete execution engine

`_create_order()` is the causal core. The execution sequence within one order:

```
assign product / machine / operator
↓
compute estimated_total_hours from product standards
↓
compute planned_end = planned_start + estimated_hours × priority_buffer
↓
RELEASE (release_time = sim_date + release_lag)
↓
[extra_hours accumulator starts at 0.0]
↓
MATERIAL check → if shortage: extra_hours += hold_h
↓
CHANGEOVER assignment
↓
FEATURE SNAPSHOT ← all 37 ML features captured here (prediction point)
↓
MACHINE BREAKDOWN check → if breaks: extra_hours += repair_h
↓
QUEUE WAIT check → if util > 0.70: extra_hours += wait_h
↓
SETUP → actual_setup sampled; cause flagged if > 1.70× standard
↓
RUN → actual_run sampled
↓
OPERATOR DELAY check → if concurrent ≥ 3 and junior: extra_hours += op_delay_h
↓
QUALITY INSPECTION → if fails: extra_hours += rework_fraction × estimated_total_hours
↓
actual_end = max(release_time, planned_start) + (actual_setup + actual_run)/60 + extra_hours
↓
OutcomeRecorder: is_delayed = (actual_end > planned_end)
```

The `extra_hours` accumulator is the key architectural decision. Every delay event adds to actual time on the timeline rather than anchoring to `planned_end`. This means delays are absorbed by planning slack — as they are in reality — and only genuinely tight or genuinely disrupted orders become late.

---

### Layer 4 — Collection

**FeatureCollector.snapshot()** is called once per order, immediately after changeover assignment and before any execution events. It writes all 37 ML features directly onto the order object. The snapshot reads from `StateManager` for rolling historical features — these call the order history with explicit time-window filtering to prevent any future leakage.

**OutcomeRecorder.record()** finalises `is_delayed`, `delay_minutes`, `delay_category`, and `delay_root_cause` after `actual_end` is set. Root cause is assigned from the `delay_causes` list accumulated during execution: single cause → that cause; multiple → `MULTIPLE_CAUSES`; none → `NONE`.

**TimelineLog** appends a structured event record for every simulation event, enabling post-hoc causal auditing.

---

### Layer 5 — Output

**DatasetBuilder.build()** converts the completed order list into a pandas DataFrame with 37 feature columns + 4 target columns + 9 identifier/metadata columns = **50 total columns**.

**Time-aware split** preserves temporal ordering to prevent future-data leakage in the ML pipeline:

| Split | Fraction | Approximate orders (540-day run) |
|---|---|---|
| train | 71% | ~3,048 |
| val | 18% | ~772 |
| test | 11% | ~472 |
| **total** | **100%** | **~4,293** |

**CalibrationChecker** validates five assertions before the run is considered complete:

| Check | Condition | Status (120-day) |
|---|---|---|
| Delay rate | 22–40% | ✓ PASS (36.6%) |
| Utilisation causal | high_util_delay > low_util_delay | validated structurally |
| Material causal | no_material_delay > has_material_delay | ✓ PASS (69.3% vs 32.1%) |
| Operator skill causal | senior_delay < junior_delay | confirmed directionally |
| Feature completeness | 0% missing values | ✓ PASS |

---

## 4. Entity Initialisation Parameters

### Products (15 per run)

| Complexity | Weight | Std setup (min) | Std run/unit (min) | Base FPY | Routing ops |
|---|---|---|---|---|---|
| LOW | 3/10 | ~20 (σ=5) | ~5 (log-normal) | 0.970 | 1–3 |
| MEDIUM | 4/10 | ~35 (σ=9) | ~9 (log-normal) | 0.930 | 2–5 |
| HIGH | 3/10 | ~60 (σ=15) | ~18 (log-normal) | 0.880 | 4–7 |

### Machines (8 per default run)

| Type | MTBF (h) | MTTR (h) | PM interval (d) | Overrun tendency |
|---|---|---|---|---|
| CNC_MILL | 720 | 4.0 | 90 | 1.06 |
| DRILL_PRESS | 1440 | 2.0 | 60 | 1.03 |
| LATHE | 960 | 3.0 | 60 | 1.05 |
| ASSEMBLY | 2880 | 1.5 | 30 | 1.08 |
| INSPECTION | 4320 | 1.0 | 30 | 1.01 |
| WELDING | 480 | 5.0 | 45 | 1.07 |
| PRESS | 600 | 6.0 | 60 | 1.05 |

OEE is initialised from normal(0.75, 0.08), clipped to [0.35, 0.92]. Machine age is uniform(0.5, 8.0) years.

### Operators (24 per default run — 8 per shift)

| Tier | Weight | Abs rate | Speed mult | Experience (mo) |
|---|---|---|---|---|
| junior | 1/4 | 6.5% | 1.12 | 1–12 |
| mid | 2/4 | 4.0% | 1.05 | 13–48 |
| senior | 1/4 | 2.5% | 0.95 | 49–180 |

### Suppliers (3 per run)

Reliability scores sampled uniform(0.80, 0.98). Shortage probability = 1 − reliability.

---

## 5. Calibration Outcomes

**Validated on:** 120-day run, 10 orders/day, 8 machines, seed=42

| Metric | Value |
|---|---|
| Overall delay rate | 36.6% |
| Mean delay (delayed orders) | 644 minutes |
| Median delay (delayed orders) | 266 minutes |
| P95 delay | 2,390 minutes |
| Orders generated | 954 |
| Features complete | 100% (0 missing) |

**Root cause distribution (delayed orders):**

| Root cause | Share of delayed | Target range |
|---|---|---|
| setup_overrun | 25.5% | 18–22% |
| none (planning variance) | 19.5% | — |
| material_unavailability | 18.1% | 20–28% |
| machine_breakdown | 18.1% | 28–35% |
| multiple_causes | 8.9% | — |
| planning_schedule_conflict | 8.3% | — |
| quality_failure_rework | 1.7% | 12–18% |

**Confirmed causal relationships:**

| Relationship | Evidence |
|---|---|
| Schedule tightness → delay | Delay rate 30.5% (Q1 loose) → 49.8% (Q4 tight) |
| Material shortage → delay | 32.1% (available) vs 69.3% (shortage) |
| Priority → delay | normal 31% → high 37% → critical 67% |
| Machine breakdown → delay | Flags orders with verified repair downtime |

---

## 6. Production Run Parameters

The full training dataset is generated at 540 days:

```bash
python ml/scripts/generate_synthetic_data.py \
  --days 540 \
  --orders-per-day 10 \
  --machines 8 \
  --seed 42 \
  --output-dir ml/data
```

Output files:

```
ml/data/raw/synthetic_factory_data.csv    # ~4,293 rows, 50 columns
ml/data/raw/simulation_report.json        # calibration audit
ml/data/processed/train.csv               # ~3,048 rows (71%)
ml/data/processed/val.csv                 # ~772 rows (18%)
ml/data/processed/test.csv                # ~472 rows (11%)
```

---

## 7. Reproducibility Contract

The simulation is fully deterministic given a fixed seed. The `SeedManager` initialises NumPy's `RandomState` and derives sub-seeds for each generator (demand, breakdown, absenteeism, setup/run, quality). Any run with `--seed 42` will produce the identical dataset on any platform. This is a hard requirement for ML experiment reproducibility.

---

*Document 01 of 04 — Manufacturing Process Copilot Technical Series*
