#!/usr/bin/env python3
"""
backend/scripts/eval_copilot.py
================================
Reproducible reliability harness for the Manufacturing Process Copilot.

Reproduces the runtime evaluation methodology documented in
docs/11_evaluation.md: send live chat requests for a fixed set of supervisor
intents, classify each response, and report aggregate reliability metrics.

Two tiers of metrics are produced:

  Client-observable (always available — no backend access needed):
    - success rate          (non-fallback, non-empty answer)
    - fallback rate         (the agent's generic "couldn't retrieve" message)
    - hallucination signal  (heuristic — see LIMITATIONS in docs/13)
    - latency               (p50 / p95 wall-clock per request)

  Log-derived (requires --docker-container — reproduces docs/11 exactly):
    - tool dispatch rate    (runs that actually called a tool)
    - average iterations    (ReAct rounds to FINAL_ANSWER)
    - answered-without-tool (the precise hallucination signal from docs/11)

The harness sends requests sequentially (with a delay) so that, when log
correlation is enabled, runs do not interleave in the backend log stream and
can be segmented by the per-turn "compress start for session <prefix>" marker.

Dependencies: Python 3.10+ standard library only (urllib, argparse, csv, json,
subprocess). No third-party packages required.

This script does NOT modify agent behavior, prompts, or model configuration.
It only observes the running system.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Default intents — the 8 supervisor intents from docs/11_evaluation.md
# ---------------------------------------------------------------------------

DEFAULT_INTENTS: list[str] = [
    "What orders are at high risk right now?",
    "Give me the current risk summary.",
    "Summarise today's shift performance.",
    "What are the current bottlenecks?",
    "Show active production orders.",
    "Give me facility KPIs.",
    "Explain delay risk for ORD-20260601-001.",
    "Why is this order predicted to be delayed?",
]

# The agent's fallback string (backend/app/services/agent/agent.py :: _FALLBACK).
# A run whose answer starts with this is classified as a fallback.
FALLBACK_PREFIX = "I wasn't able to retrieve the information needed to answer your question"

# Order-id shape used by the seed data, e.g. ORD-20260601-001.
ORDER_ID_RE = re.compile(r"ORD-\d{8}-\d{3}")

# Backend log markers (backend/app/services/agent/agent.py).
LOG_TURN_START_RE = re.compile(r"compress start for session ([0-9a-f]{8})")
LOG_DISPATCH_RE = re.compile(r"dispatching tool '([^']+)'")
LOG_FINAL_ITER_RE = re.compile(r"FINAL_ANSWER reached at iteration (\d+)")


# ---------------------------------------------------------------------------
# Per-run record
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    intent_index: int
    intent: str
    session_token: str
    status_code: int
    latency_ms: float
    answer: str
    # client-observed classification
    is_fallback: bool = False
    is_error: bool = False
    contains_order_ids: bool = False
    order_ids_in_answer: list[str] = field(default_factory=list)
    order_ids_in_prompt: list[str] = field(default_factory=list)
    # log-derived (None until/unless logs are correlated)
    tools_dispatched: list[str] | None = None
    num_dispatches: int | None = None
    final_iteration: int | None = None

    @property
    def answered_without_tool(self) -> bool | None:
        """Precise hallucination signal (docs/11): a substantive answer with no
        tool call. Requires log correlation; None if logs unavailable."""
        if self.num_dispatches is None:
            return None
        return (self.num_dispatches == 0) and not self.is_fallback and bool(self.answer.strip())


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def send_message(base_url: str, message: str, session_token: str, timeout: float) -> tuple[int, str, float]:
    """POST one chat message (stream=False). Returns (status, answer, latency_ms)."""
    url = base_url.rstrip("/") + "/api/v1/chat/message"
    body = json.dumps(
        {"message": message, "session_token": session_token, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except Exception as exc:  # noqa: BLE001 — network/timeout, recorded as status 0
        return 0, f"__REQUEST_ERROR__: {exc}", (time.perf_counter() - start) * 1000.0
    latency_ms = (time.perf_counter() - start) * 1000.0
    try:
        answer = json.loads(raw).get("content", "")
    except json.JSONDecodeError:
        answer = raw
    return status, answer, latency_ms


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(run: RunResult) -> None:
    """Populate client-observable classification fields in place."""
    answer = run.answer or ""
    run.is_error = run.status_code == 0 or run.status_code >= 500 or answer.startswith("__REQUEST_ERROR__")
    run.is_fallback = answer.strip().startswith(FALLBACK_PREFIX)
    run.order_ids_in_answer = sorted(set(ORDER_ID_RE.findall(answer)))
    run.order_ids_in_prompt = sorted(set(ORDER_ID_RE.findall(run.intent)))
    # "Contains order IDs that the user did not mention" — a weak hallucination
    # heuristic. See docs/13 LIMITATIONS: legitimate when the DB actually holds
    # those orders. The precise signal is `answered_without_tool` (needs logs).
    novel_ids = set(run.order_ids_in_answer) - set(run.order_ids_in_prompt)
    run.contains_order_ids = bool(novel_ids)


# ---------------------------------------------------------------------------
# Log correlation (optional — reproduces docs/11 methodology)
# ---------------------------------------------------------------------------

def correlate_with_docker_logs(
    container: str, runs: list[RunResult], since_seconds: int
) -> None:
    """Segment backend logs by per-turn 'compress start' markers and attach
    tool-dispatch / iteration data to each run by session-token prefix.

    Runs are issued sequentially, so log turns appear in the same order as
    `runs`. We match by the 8-char session prefix the agent logs.
    """
    try:
        out = subprocess.run(
            ["docker", "logs", container, "--since", f"{since_seconds}s"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not read docker logs for '{container}': {exc}", file=sys.stderr)
        return

    lines = (out.stdout + out.stderr).splitlines()

    # Build per-turn segments keyed by session prefix, preserving order.
    segments: dict[str, list[dict]] = {}
    order: list[str] = []
    cur: str | None = None
    for line in lines:
        m = LOG_TURN_START_RE.search(line)
        if m:
            cur = m.group(1)
            segments.setdefault(cur, [])
            order.append(cur)
            segments[cur].append({"tools": [], "final_iter": None})
            continue
        if cur is None:
            continue
        seg = segments[cur][-1]
        dm = LOG_DISPATCH_RE.search(line)
        if dm:
            seg["tools"].append(dm.group(1))
            continue
        fm = LOG_FINAL_ITER_RE.search(line)
        if fm:
            seg["final_iter"] = int(fm.group(1))

    # Attach to runs by prefix. If a prefix recurs (rare), consume in order.
    consumed: dict[str, int] = {}
    for run in runs:
        prefix = run.session_token[:8]
        bucket = segments.get(prefix)
        if not bucket:
            continue
        idx = consumed.get(prefix, 0)
        if idx >= len(bucket):
            continue
        seg = bucket[idx]
        consumed[prefix] = idx + 1
        run.tools_dispatched = list(seg["tools"])
        run.num_dispatches = len(seg["tools"])
        run.final_iteration = seg["final_iter"]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(runs: list[RunResult], have_logs: bool) -> dict:
    total = len(runs)
    by_intent: dict[int, list[RunResult]] = {}
    for r in runs:
        by_intent.setdefault(r.intent_index, []).append(r)

    def rate(predicate) -> float:
        return sum(1 for r in runs if predicate(r)) / total if total else 0.0

    latencies = [r.latency_ms for r in runs if not r.is_error]
    summary = {
        "total_runs": total,
        "success_rate": rate(lambda r: not r.is_fallback and not r.is_error and bool(r.answer.strip())),
        "fallback_rate": rate(lambda r: r.is_fallback),
        "error_rate": rate(lambda r: r.is_error),
        "contains_novel_order_ids_rate": rate(lambda r: r.contains_order_ids),
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "latency_p95_ms": round(_percentile(latencies, 95), 1) if latencies else None,
    }

    if have_logs:
        dispatched = [r for r in runs if r.num_dispatches is not None]
        if dispatched:
            summary["tool_dispatch_rate"] = sum(1 for r in dispatched if r.num_dispatches > 0) / len(dispatched)
            iters = [r.final_iteration for r in dispatched if r.final_iteration is not None]
            summary["avg_iterations"] = round(statistics.mean(iters), 2) if iters else None
            summary["hallucination_rate"] = sum(1 for r in dispatched if r.answered_without_tool) / len(dispatched)
            summary["log_correlated_runs"] = len(dispatched)
        else:
            summary["tool_dispatch_rate"] = None
            summary["avg_iterations"] = None
            summary["hallucination_rate"] = None
            summary["log_correlated_runs"] = 0
    else:
        summary["tool_dispatch_rate"] = None
        summary["avg_iterations"] = None
        summary["hallucination_rate"] = None
        summary["log_correlated_runs"] = 0

    # Per-intent breakdown
    per_intent = []
    for idx in sorted(by_intent):
        group = by_intent[idx]
        n = len(group)
        entry = {
            "intent_index": idx,
            "intent": group[0].intent,
            "n": n,
            "fallback_rate": sum(1 for r in group if r.is_fallback) / n,
            "success_rate": sum(1 for r in group if not r.is_fallback and not r.is_error and r.answer.strip()) / n,
        }
        if have_logs:
            disp = [r for r in group if r.num_dispatches is not None]
            if disp:
                tool_counts: dict[str, int] = {}
                for r in disp:
                    for t in (r.tools_dispatched or []):
                        tool_counts[t] = tool_counts.get(t, 0) + 1
                iters = [r.final_iteration for r in disp if r.final_iteration is not None]
                entry["tools"] = tool_counts
                entry["avg_iterations"] = round(statistics.mean(iters), 2) if iters else None
        per_intent.append(entry)
    summary["per_intent"] = per_intent
    return summary


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def print_summary(summary: dict, have_logs: bool) -> None:
    print("\n" + "=" * 70)
    print("COPILOT RELIABILITY SUMMARY")
    print("=" * 70)
    print(f"Total runs:              {summary['total_runs']}")
    print(f"Success rate:            {summary['success_rate']:.1%}")
    print(f"Fallback rate:           {summary['fallback_rate']:.1%}")
    print(f"Error rate:              {summary['error_rate']:.1%}")
    print(f"Latency p50 / p95 (ms):  {summary['latency_p50_ms']} / {summary['latency_p95_ms']}")
    if have_logs and summary.get("log_correlated_runs"):
        print(f"Tool dispatch rate:      {summary['tool_dispatch_rate']:.1%}")
        print(f"Avg iterations:          {summary['avg_iterations']}")
        print(f"Hallucination rate:      {summary['hallucination_rate']:.1%}  (answered without a tool call)")
        print(f"(log-correlated runs:    {summary['log_correlated_runs']}/{summary['total_runs']})")
    else:
        print("Tool dispatch rate:      N/A  (pass --docker-container to enable)")
        print("Avg iterations:          N/A  (pass --docker-container to enable)")
        print("Hallucination rate:      N/A  (precise signal needs --docker-container)")
        print(f"Novel-order-id rate:     {summary['contains_novel_order_ids_rate']:.1%}  (weak heuristic — see docs/13)")
    print("-" * 70)
    print("Per-intent:")
    for e in summary["per_intent"]:
        line = f"  [{e['intent_index']}] {e['intent'][:42]:<42} n={e['n']} " \
               f"ok={e['success_rate']:.0%} fb={e['fallback_rate']:.0%}"
        if "avg_iterations" in e and e["avg_iterations"] is not None:
            line += f" iter={e['avg_iterations']}"
        print(line)
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(path: str, summary: dict, runs: list[RunResult], meta: dict) -> None:
    payload = {"meta": meta, "summary": summary, "runs": [asdict(r) for r in runs]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[ok] wrote JSON: {path}")


def write_csv(path: str, runs: list[RunResult]) -> None:
    cols = [
        "intent_index", "intent", "session_token", "status_code", "latency_ms",
        "is_fallback", "is_error", "contains_order_ids",
        "num_dispatches", "tools_dispatched", "final_iteration", "answered_without_tool",
        "answer",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in runs:
            w.writerow([
                r.intent_index, r.intent, r.session_token, r.status_code,
                round(r.latency_ms, 1), r.is_fallback, r.is_error, r.contains_order_ids,
                r.num_dispatches,
                "|".join(r.tools_dispatched) if r.tools_dispatched else "",
                r.final_iteration, r.answered_without_tool,
                (r.answer or "").replace("\n", " ")[:500],
            ])
    print(f"[ok] wrote CSV:  {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Reproducible reliability harness for the MPC copilot (see docs/13).",
    )
    p.add_argument("--base-url", default="http://localhost:8000",
                   help="Backend base URL (default: http://localhost:8000)")
    p.add_argument("--runs", type=int, default=10,
                   help="Runs per intent (default: 10 — matches docs/11)")
    p.add_argument("--intents-file", default=None,
                   help="Optional path to a text file with one intent per line "
                        "(default: the 8 intents from docs/11)")
    p.add_argument("--delay", type=float, default=1.5,
                   help="Seconds to wait between requests (default: 1.5)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-request timeout in seconds (default: 120)")
    p.add_argument("--docker-container", default=None,
                   help="If set, correlate backend logs from this container to "
                        "compute tool-dispatch rate and avg iterations "
                        "(e.g. manufacturing-process-copilot-backend-1)")
    p.add_argument("--json-out", default=None, help="Path to write full JSON results")
    p.add_argument("--csv-out", default=None, help="Path to write per-run CSV")
    args = p.parse_args()

    if args.intents_file:
        with open(args.intents_file, encoding="utf-8") as fh:
            intents = [ln.strip() for ln in fh if ln.strip()]
    else:
        intents = DEFAULT_INTENTS

    total_planned = len(intents) * args.runs
    print(f"Evaluating {len(intents)} intents x {args.runs} runs = {total_planned} requests")
    print(f"Target: {args.base_url}")

    runs: list[RunResult] = []
    started = time.time()
    for idx, intent in enumerate(intents):
        for _ in range(args.runs):
            token = str(uuid.uuid4())
            status, answer, latency = send_message(args.base_url, intent, token, args.timeout)
            run = RunResult(
                intent_index=idx, intent=intent, session_token=token,
                status_code=status, latency_ms=latency, answer=answer,
            )
            classify(run)
            runs.append(run)
            tag = "ERR" if run.is_error else ("FB" if run.is_fallback else "OK")
            print(f"  [{idx}] {tag:<3} {latency:7.0f}ms  {answer[:60].replace(chr(10),' ')}")
            time.sleep(args.delay)
        print(f"  intent {idx} done")

    elapsed = time.time() - started
    have_logs = bool(args.docker_container)
    if have_logs:
        # +60s margin so the first run's log lines fall inside the window.
        correlate_with_docker_logs(args.docker_container, runs, int(elapsed) + 60)

    summary = summarize(runs, have_logs)
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "runs_per_intent": args.runs,
        "intents": intents,
        "log_correlation": have_logs,
        "elapsed_seconds": round(elapsed, 1),
    }

    print_summary(summary, have_logs)
    if args.json_out:
        write_json(args.json_out, summary, runs, meta)
    if args.csv_out:
        write_csv(args.csv_out, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
