# Manufacturing Process Copilot
## Document 03 — Final Feature Dictionary

**Status:** Production-Ready  
**Version:** 1.0  
**Total features:** 37  
**Prediction point:** Order release — all features reflect knowable state before execution begins

---

## Reading this Document

Each feature entry contains:

- **Type** — pandas dtype in the output CSV
- **Range** — [min, max] observed in the 120-day validation run
- **Mean / Median** — summary statistics
- **|r| with is_delayed** — absolute Pearson correlation with the primary target
- **Causal role** — whether this feature causally produces delays in the simulation, or is a proxy/historical signal
- **ML notes** — preprocessing, expected importance, interactions

---

## Category A — Order Planning (8 features)

These features describe the production order's planning parameters as known at release time. They are among the strongest predictors because they directly determine how much slack exists between estimated work and committed deadline.

---

### `planned_lead_time_hours`
| | |
|---|---|
| Type | float64 |
| Unit | Hours |
| Range | [8.3, 224.8] |
| Mean / Median | 30.2 / 26.8 |
| **\|r\| with is_delayed** | **0.257** |
| Causal role | Proxy (longer lead time → more buffer → less delay) |

The total planned time from order creation to completion deadline. Computed as `(planned_end − sim_date)` in hours. Longer lead times correspond to lower-priority orders with more planning slack. Negatively correlated with delay when controlled for job size; positively correlated marginally because larger/harder jobs also get longer lead times. Apply log1p transform.

---

### `release_lag_hours`
| | |
|---|---|
| Type | float64 |
| Unit | Hours |
| Range | [0.8, 140.0] |
| Mean / Median | 12.0 / 7.0 |
| **\|r\| with is_delayed** | **0.478** (2nd highest) |
| Causal role | Causal (long release lag eats into available processing window) |

Hours between order creation and actual material release. An order released 40 hours late on a 50-hour window has almost no chance of completing on time regardless of execution quality. This is the single most powerful planning-time predictor. Apply log1p transform. Strong interaction with `schedule_tightness_ratio`.

---

### `schedule_revision_count`
| | |
|---|---|
| Type | float64 |
| Unit | Count |
| Range | [0, 1] |
| Mean / Median | 0.031 / 0.0 |
| **\|r\| with is_delayed** | ~0.09 |
| Causal role | Proxy (expedited/revised orders signal upstream scheduling stress) |

Number of times the order was revised before release. Currently 0 or 1 (1 = expedited critical order with revision). Low frequency (3.1% of orders). Treat as binary flag in practice.

---

### `is_expedited`
| | |
|---|---|
| Type | int64 |
| Unit | Binary (0/1) |
| Range | {0, 1} |
| Mean / Median | 0.031 / 0 |
| **\|r\| with is_delayed** | 0.088 |
| Causal role | Proxy (expedited orders are critical-priority with reduced buffer) |

Flag set for ~35% of critical-priority orders when they are explicitly expedited. Rare overall (3.1%). Correlated with high delay rate because expedited orders have the tightest buffers (1.15×), not because being expedited itself causes delays.

---

### `priority_encoded`
| | |
|---|---|
| Type | int64 |
| Encoding | 1=normal, 2=high, 3=critical |
| Range | {1, 2, 3} |
| Mean / Median | 1.53 / 1.0 |
| **\|r\| with is_delayed** | 0.194 |
| Causal role | Causal (determines planned_end buffer size → directly affects slack) |

Ordinal-encoded priority. Higher priority = tighter planning buffer = higher delay probability. Delay rates: normal 31%, high 37%, critical 67%. Use as ordinal integer; no one-hot encoding needed for tree models.

---

### `quantity`
| | |
|---|---|
| Type | int64 |
| Unit | Units |
| Range | [3, 617] |
| Mean / Median | 45.8 / 33 |
| **\|r\| with is_delayed** | 0.169 |
| Causal role | Causal (higher quantity → higher run time → harder to complete within window) |

Order quantity. Log-normally distributed (lognormal(3.5, 0.8) → median ~33). Drives `estimated_total_hours` and thereby `schedule_tightness_ratio`. Apply log1p transform. Do not confuse with `material_bom_complexity`, which is a product property.

---

### `estimated_total_hours`
| | |
|---|---|
| Type | float64 |
| Unit | Hours |
| Range | [0.70, 115.1] |
| Mean / Median | 9.6 / 6.5 |
| **\|r\| with is_delayed** | 0.191 |
| Causal role | Proxy (numerator of tightness_ratio; large jobs have less buffer per planned hour) |

Standard setup time + standard run time × quantity, in hours. This is the planner's best estimate of required processing time. Apply log1p transform. Directly related to `quantity` × `product_complexity_score`. The ratio `estimated_total_hours / planned_window` = `schedule_tightness_ratio`.

---

### `schedule_tightness_ratio`
| | |
|---|---|
| Type | float64 |
| Unit | Ratio (dimensionless) |
| Range | [0.18, 1.02] |
| Mean / Median | 0.60 / 0.59 |
| **\|r\| with is_delayed** | 0.185 |
| Causal role | Causal (ratio determines how much variance the planned window can absorb) |

`estimated_total_hours ÷ planned_window_hours`. The fundamental scheduling risk metric. A ratio of 0.90 means only 11% slack — any overrun of >11% on processing time will delay the order. Delay rate by quartile: Q1 (0.18–0.53) = 30.5%, Q4 (0.65–1.02) = 49.8%. No transform needed; already bounded [0,1].

---

## Category B — Product Characteristics (3 features)

These describe the inherent properties of the product being manufactured. They are known at order creation (not just release). They are moderate predictors because complexity correlates with all failure modes simultaneously.

---

### `product_complexity_score`
| | |
|---|---|
| Type | float64 |
| Unit | Score (0.25–0.85 scale) |
| Range | [0.25, 0.85] |
| Mean / Median | 0.59 / 0.55 |
| **\|r\| with is_delayed** | 0.120 |
| Causal role | Proxy (complexity drives setup time, run time, and FPY — all indirect) |

Normalised complexity score derived from `ProductComplexity` enum: LOW=0.25, MEDIUM=0.55, HIGH=0.85. Correlates with longer setup times, lower FPY, and more operations. Use as continuous feature; no transform needed.

---

### `material_bom_complexity`
| | |
|---|---|
| Type | int64 |
| Unit | Count (BOM line items) |
| Range | [2, 9] |
| Mean / Median | 5.1 / 5 |
| **\|r\| with is_delayed** | ~0.06 |
| Causal role | Proxy (more BOM lines → higher probability of any single component being short) |

Number of distinct components in the bill of materials. Used to sample `component_shortage_count` when a shortage occurs. Low standalone predictive value; most signal comes through `material_availability_at_release`.

---

### `operation_count`
| | |
|---|---|
| Type | int64 |
| Unit | Count |
| Range | [2, 6] |
| Mean / Median | 3.6 / 3 |
| **\|r\| with is_delayed** | ~0.05 |
| Causal role | Proxy (more operations → more complex routing → higher inherent risk) |

Number of operations in the product's routing. LOW complexity: 1–3 ops; MEDIUM: 2–5; HIGH: 4–7. Correlated with `product_complexity_score`. Low marginal predictive value once complexity score is included.

---

## Category C — Machine State at Release (8 features)

Machine features represent the condition and workload of the assigned machine at the moment the order is released. They are causal because machine condition directly affects breakdown probability and setup performance.

---

### `machine_utilization_at_release`
| | |
|---|---|
| Type | float64 |
| Unit | Ratio (0–1) |
| Range | [0.0, 1.0] |
| Mean / Median | 0.48 / 0.36 |
| **\|r\| with is_delayed** | ~0.05 |
| Causal role | Causal (high utilisation triggers queue wait → delay) |

Fraction of scheduled time the machine was occupied in the 24 hours preceding release. Bimodal distribution: ~50% near 0 (early simulation / idle machines) and ~43% near 1 (heavily loaded machines). The causal path is: high_utilisation → queue wait in execution → higher actual_end → delay. Tree models will discover the threshold effect near 0.70.

---

### `work_center_queue_depth_at_release`
| | |
|---|---|
| Type | float64 |
| Unit | Queue depth (0–1 normalised) |
| Range | [0.0, 1.0] |
| Mean / Median | 0.69 / 1.0 |
| **\|r\| with is_delayed** | **0.502** (strongest feature) |
| Causal role | Causal (high queue → congestion → longer wait → delay) |

The strongest predictor in the dataset. Binary in practice (0 = empty queue, 1 = orders waiting) due to current queue depth implementation. High correlation with delay because orders encountering a non-empty queue must wait before starting, compressing their available window. No transform needed.

---

### `machine_oee_30d`
| | |
|---|---|
| Type | float64 |
| Unit | OEE score (0–1) |
| Range | [0.57, 0.79] |
| Mean / Median | 0.64 / 0.65 |
| **\|r\| with is_delayed** | ~0.03 |
| Causal role | Proxy (lower OEE → higher breakdown probability → higher delay in simulation) |

Overall Equipment Effectiveness over the trailing 30 days. Degrades when maintenance is overdue, recovers after PM. The causal path is indirect (OEE → breakdown probability → order delay), which is why the direct correlation is weak. The machine_delay_rate_90d feature captures the effect more directly.

---

### `machine_unplanned_downtime_hours_30d`
| | |
|---|---|
| Type | float64 |
| Unit | Hours |
| Range | [0.0, 6.6] |
| Mean / Median | 1.89 / 1.50 |
| **\|r\| with is_delayed** | ~0.04 |
| Causal role | Proxy (recent breakdown history predicts future breakdown probability) |

Estimated unplanned downtime on this machine in the trailing 30 days. Right-skewed; apply log1p transform. Correlated with `machine_delay_rate_90d` — use both since downtime captures severity while delay rate captures frequency.

---

### `days_since_last_planned_maintenance`
| | |
|---|---|
| Type | float64 |
| Unit | Days |
| Range | [0, 89] |
| Mean / Median | 24.1 / 22.0 |
| **\|r\| with is_delayed** | ~0.03 |
| Causal role | Causal (higher value → higher breakdown probability → higher delay) |

Days since the machine last had planned maintenance. OEE degrades and breakdown probability increases as this value grows relative to the PM interval. Combine with `maintenance_due_within_order_window` for interaction effects.

---

### `maintenance_due_within_order_window`
| | |
|---|---|
| Type | int64 |
| Unit | Binary (0/1) |
| Range | {0, 1} |
| Mean / Median | 0.014 / 0 |
| **\|r\| with is_delayed** | ~0.04 |
| Causal role | Proxy (PM event during order window → machine potentially unavailable) |

Flag: 1 if planned maintenance is scheduled to fall within the order's planned execution window. Very rare (1.4%). When it fires, the order risks disruption from the PM event. Useful as a rare but high-information flag.

---

### `changeover_required`
| | |
|---|---|
| Type | int64 |
| Unit | Binary (0/1) |
| Range | {0, 1} |
| Mean / Median | 0.51 / 1.0 |
| **\|r\| with is_delayed** | ~0.02 |
| Causal role | Proxy (changeover inflates actual setup time) |

1 if the order requires a tooling/fixture changeover because the previous product on this machine was different. Fires for ~51% of orders. Amplified by `changeover_complexity_score`. Direct causal path: changeover → higher actual_setup → more likely to exceed planning buffer.

---

### `changeover_complexity_score`
| | |
|---|---|
| Type | float64 |
| Unit | Multiplier (1.0–3.0) |
| Range | [1.0, 3.0] |
| Mean / Median | 1.65 / 1.55 |
| **\|r\| with is_delayed** | 0.102 |
| Causal role | Causal (higher complexity → setup_speed_multiplier applied × complexity → longer actual setup) |

When `changeover_required = 1`, this is the complexity multiplier applied to base setup time. Sampled uniform(1.5, 3.0) when changeover fires; fixed at 1.0 when it does not. Interaction with `changeover_required` is important — always include both features.

---

## Category D — Operator State at Release (5 features)

Operator features capture the human factor at execution time. They are moderate predictors individually but important for the `delay_root_cause` model, particularly for identifying operator-attributed delays.

---

### `operator_experience_months`
| | |
|---|---|
| Type | int64 |
| Unit | Months |
| Range | [1, 176] |
| Mean / Median | 59.3 / 34 |
| **\|r\| with is_delayed** | ~0.04 |
| Causal role | Proxy (experience determines skill tier → setup speed → actual processing time) |

Raw experience in months. Right-skewed (junior operators dominate the low end). The skill tier encoding is a cleaner feature for most models; keep both since high experience within a tier can matter. Apply log1p transform.

---

### `operator_skill_tier_encoded`
| | |
|---|---|
| Type | float64 |
| Encoding | 0=junior, 1=mid, 2=senior |
| Range | {0.0, 1.0, 2.0} |
| Mean / Median | 1.06 / 1.0 |
| **\|r\| with is_delayed** | ~0.03 |
| Causal role | Causal (tier determines setup_speed_multiplier → actual_setup → delay probability) |

The primary operator capability signal. Junior operators (1.12× setup time), mid (1.05×), senior (0.95×). Delay rates: junior 33.6%, mid 38.3%, senior 33.8%. The mid tier's higher delay rate reflects their frequency in high-load situations. Treat as ordinal; no one-hot encoding needed.

---

### `operator_concurrent_order_count`
| | |
|---|---|
| Type | float64 |
| Unit | Count |
| Range | [0, 0] |
| Mean / Median | 0.0 / 0.0 |
| **\|r\| with is_delayed** | 0.0 |
| Causal role | Planned causal; currently zero in single-shift model |
| **Status** | Placeholder — will be non-zero in multi-machine operator scenarios |

The count of other orders the operator is actively running in parallel. Zero in the current simulation because each operator is single-assigned per order. Preserved as a feature placeholder for the extended model where operators manage multiple machines simultaneously. Set to zero and retain in the feature set; the model will learn its zero-variance and ignore it.

---

### `hours_into_shift_at_start`
| | |
|---|---|
| Type | float64 |
| Unit | Hours |
| Range | [0.0, 7.5] |
| Mean / Median | 3.4 / 3.0 |
| **\|r\| with is_delayed** | ~0.03 |
| Causal role | Proxy (fatigue effect — run time increases 2.5% per hour beyond 4h into shift) |

How many hours into the operator's shift the order starts. Operators working in the final 3 hours of their shift have a small fatigue multiplier on run time. Uniform-ish distribution across [0, 7.5].

---

### `shift_type_encoded`
| | |
|---|---|
| Type | int64 |
| Encoding | 0=morning, 1=afternoon, 2=night |
| Range | {0, 1, 2} |
| Mean / Median | 0.79 / 1.0 |
| **\|r\| with is_delayed** | ~0.06 |
| Causal role | Proxy (shift affects absenteeism and fatigue base rates) |

Shift assignment of the operator. Night shift operators have slightly higher absenteeism. The `shift_delay_rate_30d` historical feature captures the shift-level performance more directly.

---

## Category E — Material State at Release (2 features)

Material features are among the strongest predictors because they directly cause order holds in the simulation.

---

### `material_availability_at_release`
| | |
|---|---|
| Type | int64 |
| Unit | Binary (0/1) |
| Range | {0, 1} |
| Mean / Median | 0.88 / 1.0 |
| **\|r\| with is_delayed** | 0.250 |
| Causal role | Causal (0 → material hold → extra_hours added → likely delay) |

1 = all materials available at release; 0 = component shortage, order held. ~11.8% of orders experience a shortage. When shortage occurs, delay rate is 69.3% vs 32.1% for available orders — the largest absolute gap of any feature. The single most operationally actionable predictor: proactive material checks before release can directly reduce delay rates.

---

### `component_shortage_count`
| | |
|---|---|
| Type | float64 |
| Unit | Count |
| Range | [0, 2] |
| Mean / Median | 0.16 / 0.0 |
| **\|r\| with is_delayed** | 0.226 |
| Causal role | Causal (more missing components → longer hold duration on average) |

Number of distinct components that are in shortage. 0 when `material_availability_at_release = 1`; 1 or 2 when shortage. Captures severity of the shortage. Use alongside `material_availability_at_release` — they are complementary.

---

## Category F — Historical Rolling Features (7 features)

These are the most sophisticated features. They are computed by querying the simulation's completed order history at the snapshot time, with strict temporal windowing. They encode accumulated evidence about the reliability of the specific product-machine-operator combination — the kind of tacit knowledge an experienced production planner carries.

All rolling features default to a population mean when fewer than 3 historical orders exist (cold start). This means early-simulation orders will have less informative values; the ML model should treat these gracefully (tree models do this naturally).

---

### `product_delay_rate_90d`
| | |
|---|---|
| Type | float64 |
| Unit | Rate (0–1) |
| Range | [0.0, 1.0] |
| Mean / Median | 0.343 / 0.321 |
| **\|r\| with is_delayed** | 0.113 |
| Causal role | Predictive proxy (past delays predict future delays for same product) |

Fraction of completed orders for this product in the trailing 90 days that were delayed. Captures intrinsic product difficulty not fully explained by complexity score alone. Strong cold-start sensitivity; plan for ~30% of the train set having default values.

---

### `machine_delay_rate_90d`
| | |
|---|---|
| Type | float64 |
| Unit | Rate (0–1) |
| Range | [0.08, 0.70] |
| Mean / Median | 0.347 / 0.358 |
| **\|r\| with is_delayed** | ~0.07 |
| Causal role | Predictive proxy (machine reliability history predicts breakdown-caused delays) |

Fraction of orders on this machine in the trailing 90 days that were delayed. Captures accumulated machine performance including breakdown history and OEE degradation. Less noisy than `product_delay_rate_90d` because machines process many more orders than a single product.

---

### `operator_delay_rate_90d`
| | |
|---|---|
| Type | float64 |
| Unit | Rate (0–1) |
| Range | [0.0, 0.67] |
| Mean / Median | 0.354 / 0.351 |
| **\|r\| with is_delayed** | ~0.05 |
| Causal role | Predictive proxy (operator performance history) |

Fraction of orders assigned to this operator in the trailing 90 days that were delayed. High variance (std=0.089) because each operator processes fewer orders than each machine. Noisy early in simulation; gains signal after ~60 simulation days.

---

### `product_x_machine_delay_rate_90d`
| | |
|---|---|
| Type | float64 |
| Unit | Rate (0–1) |
| Range | [0.0, 1.0] |
| Mean / Median | 0.342 / 0.320 |
| **\|r\| with is_delayed** | 0.101 |
| Causal role | Predictive proxy (interaction captures machine-product compatibility issues) |

Fraction of delayed orders specifically for this product on this machine combination. Captures compatibility effects — a specific product on a specific machine may have systematic setup problems that don't show up in either margin. Very sparse early in simulation; defaults to population mean. Highly correlated with `product_delay_rate_90d` (|r|≈0.94) — include both for tree models, consider dropping one for linear models.

---

### `product_first_pass_yield_90d`
| | |
|---|---|
| Type | float64 |
| Unit | Rate (0–1) |
| Range | [0.50, 1.0] |
| Mean / Median | 0.916 / 0.929 |
| **\|r\| with is_delayed** | ~0.04 |
| Causal role | Predictive proxy (low FPY predicts quality-caused rework delays) |

First-pass yield for this product in the trailing 90 days. Low values predict quality failure events and their associated rework delays. Useful specifically for `delay_root_cause = quality_failure_rework` classification. Defaults to product's base FPY when no history exists.

---

### `machine_setup_overrun_rate_90d`
| | |
|---|---|
| Type | float64 |
| Unit | Rate (0–1) |
| Range | [0.0, 0.83] |
| Mean / Median | 0.521 / 0.669 |
| **\|r\| with is_delayed** | ~0.02 |
| Causal role | Predictive proxy (high overrun rate predicts future setup overruns) |

Fraction of orders on this machine where actual setup exceeded 1.5× standard in the trailing 90 days. Note the threshold difference: the feature uses 1.5× (a softer signal) while the root cause flag uses 1.7× (harder threshold). The historical rate is a leading indicator of current setup risk.

---

### `shift_delay_rate_30d`
| | |
|---|---|
| Type | float64 |
| Unit | Rate (0–1) |
| Range | [0.20, 0.43] |
| Mean / Median | 0.358 / 0.357 |
| **\|r\| with is_delayed** | ~0.03 |
| Causal role | Predictive proxy (shift-level performance baseline) |

Factory-wide delay rate for all orders on this shift in the trailing 30 days. Narrow range (0.20–0.43) because it's an aggregate. Useful for detecting systematic shift-level problems (e.g. night shift absenteeism surge). Low standalone predictive value.

---

## Category G — Temporal Features (4 features)

Calendar features encode demand pattern effects, seasonal quality variation, and scheduling pressure from business cycle events.

---

### `planned_start_day_of_week`
| | |
|---|---|
| Type | float64 |
| Unit | Weekday (0=Monday ... 4=Friday) |
| Range | [0.0, 4.0] |
| Mean / Median | 1.86 / 2.0 |
| **\|r\| with is_delayed** | ~0.04 |
| Causal role | Proxy (Friday orders have less recovery time if anything goes wrong) |

Day of week (Mon–Fri only; weekends skipped). Monday and Friday orders have slightly different delay patterns due to week-start rush and end-of-week overtime constraints. For tree models: keep as ordinal. For linear models: consider cyclical sin/cos encoding if consistent periodicity emerges in EDA.

---

### `planned_start_hour`
| | |
|---|---|
| Type | int64 |
| Unit | Hour (4–23) |
| Range | [4, 23] |
| Mean / Median | 13.7 / 13 |
| **\|r\| with is_delayed** | 0.118 |
| Causal role | Proxy (late-shift starts have higher fatigue exposure and shift boundary risk) |

Hour of day when the order is planned to start. Orders starting late in the afternoon shift (18:00–22:00) or on night shift are more exposed to shift boundary interruptions. Moderate predictor. Consider cyclical encoding for neural networks.

---

### `is_month_end`
| | |
|---|---|
| Type | int64 |
| Unit | Binary (0/1) |
| Range | {0, 1} |
| Mean / Median | 0.227 / 0 |
| **\|r\| with is_delayed** | ~0.02 |
| Causal role | Proxy (EOM demand surge increases factory load and scheduling pressure) |

Flag: 1 in the last 4 working days of the calendar month (22.7% of orders). `DemandGenerator` applies a 1.35× order rate multiplier during these days, creating queue congestion. Low standalone signal; predictive value emerges in interaction with `machine_utilization_at_release` and `work_center_queue_depth_at_release`.

---

### `is_quarter_end`
| | |
|---|---|
| Type | int64 |
| Unit | Binary (0/1) |
| Range | {0, 1} |
| Mean / Median | 0.096 / 0 |
| **\|r\| with is_delayed** | ~0.01 |
| Causal role | Proxy (EOQ demand surge creates peak scheduling pressure) |

Flag: 1 in the last 4 working days of Q1/Q2/Q3/Q4 (9.6% of orders). `DemandGenerator` applies a 1.70× order rate multiplier during these days — the strongest demand spike in the simulation. Quarter-end days are always also month-end days (`is_month_end = 1` whenever `is_quarter_end = 1`); do not sum these features. Use `is_quarter_end` independently to isolate the highest-congestion periods. Lowest standalone predictor in the dataset but useful in conjunction with `machine_utilization_at_release` and `work_center_queue_depth_at_release`.

---

## Feature Priority Summary

Ranked by confirmed predictive value and causal clarity for the ML pipeline:

| Tier | Features | Rationale |
|---|---|---|
| **Tier 1 — Must-have** | `work_center_queue_depth_at_release`, `release_lag_hours`, `material_availability_at_release`, `schedule_tightness_ratio`, `component_shortage_count`, `priority_encoded` | Highest correlations; direct causal paths in simulation |
| **Tier 2 — High value** | `planned_lead_time_hours`, `estimated_total_hours`, `product_delay_rate_90d`, `product_x_machine_delay_rate_90d`, `changeover_complexity_score`, `product_complexity_score`, `quantity`, `planned_start_hour` | Strong signal; confirmed in EDA correlation analysis |
| **Tier 3 — Contextual** | All remaining 23 features | Individually weak but collectively improve coverage of edge cases, interpretability, and root cause classification |

No feature should be dropped from the initial baseline. Feature selection, if performed, should use permutation importance on the validation split after baseline training — not pre-training heuristics.

---

## Leakage Certification

All 37 features are confirmed leakage-free by construction:

1. All features are captured by `FeatureCollector.snapshot()` at order release, before any execution begins.
2. All historical rolling features use explicit `release_time - window_days` as the lookback boundary, never querying from completed orders with timestamps after `release_time`.
3. No feature encodes knowledge of `actual_end`, `delay_minutes`, or any execution outcome.
4. `work_center_queue_depth_at_release` reflects the queue state at the snapshot time using only historical scheduled windows, not future windows.

This leakage certificate is structural: the simulator cannot produce a leaky feature by design, because features are captured on a different code path (Layer 4 collection) from outcomes (OutcomeRecorder, which runs after execution completes).

---

*Document 03 of 04 — Manufacturing Process Copilot Technical Series*
