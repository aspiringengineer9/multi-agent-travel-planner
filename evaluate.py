#!/usr/bin/env python3
"""
Evaluation harness — Phase 2c.

Runs all scenarios N times and computes:
  - Agreement rate (target: ≥89%)
  - Average rounds to resolution (coordination efficiency)
  - Tool call success rate across ReAct iterations (target: 25% improvement vs. baseline)
  - Per-scenario and per-tool breakdowns

Usage:
  python evaluate.py                     # 2 runs per scenario
  python evaluate.py --runs 3            # 3 runs per scenario
  python evaluate.py --report-only       # read existing output files, no new runs
  python evaluate.py --scenario barcelona  # run one scenario only
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from negotiation_graph import run_session

SCENARIOS = {
    "barcelona":       "scenarios/scenario_barcelona.yaml",
    "la":              "scenarios/scenario_la.yaml",
    "family_vacation": "scenarios/scenario_family_vacation.yaml",
    "business_trip":   "scenarios/scenario_business_trip.yaml",
    "ski_trip":        "scenarios/scenario_ski_trip.yaml",
}

AGREEMENT_SCORES = {
    "consensus":         1.0,
    "partial_consensus": 0.5,
    "impasse":           0.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Running sessions
# ─────────────────────────────────────────────────────────────────────────────

def run_all(scenario_filter: str | None, runs_per_scenario: int) -> list[dict]:
    targets = (
        {scenario_filter: SCENARIOS[scenario_filter]}
        if scenario_filter and scenario_filter in SCENARIOS
        else SCENARIOS
    )
    results = []
    total = len(targets) * runs_per_scenario
    done = 0
    for name, rel_path in targets.items():
        full_path = os.path.join(SCRIPT_DIR, rel_path)
        if not os.path.exists(full_path):
            print(f"[SKIP] {rel_path} — file not found")
            continue
        for run_idx in range(runs_per_scenario):
            done += 1
            print(f"\n{'═'*64}")
            print(f"  [{done}/{total}] {name}  (run {run_idx + 1}/{runs_per_scenario})")
            print('═'*64)
            try:
                final_state = run_session(full_path)
                status = final_state.get("status", "unknown")
                rounds = final_state.get("round", 0)
                max_rounds = final_state.get("max_rounds", 3)
                results.append({
                    "scenario": name,
                    "run": run_idx + 1,
                    "status": status,
                    "agreement_score": AGREEMENT_SCORES.get(status, 0.0),
                    "rounds": rounds,
                    "max_rounds": max_rounds,
                    "early_termination": rounds < max_rounds,
                })
                print(f"\n  → {status.upper()}  ({rounds}/{max_rounds} rounds)")
            except Exception as e:
                print(f"  → ERROR: {e}")
                results.append({
                    "scenario": name,
                    "run": run_idx + 1,
                    "status": "error",
                    "agreement_score": 0.0,
                    "rounds": 0,
                    "max_rounds": 3,
                    "early_termination": False,
                })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_session_metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    total = len(results)
    agreed = sum(1 for r in results if r["status"] in ("consensus", "partial_consensus"))
    agreement_rate = agreed / total
    avg_score = sum(r["agreement_score"] for r in results) / total
    avg_rounds = sum(r["rounds"] for r in results) / total
    avg_max = sum(r["max_rounds"] for r in results) / total
    early = sum(1 for r in results if r["early_termination"])

    # Coordination efficiency: rounds saved vs. always running to max
    coord_efficiency = (avg_max - avg_rounds) / avg_max if avg_max > 0 else 0.0

    by_scenario: dict[str, dict] = defaultdict(lambda: {"runs": 0, "agreed": 0, "total_score": 0.0, "total_rounds": 0})
    for r in results:
        s = by_scenario[r["scenario"]]
        s["runs"] += 1
        s["total_score"] += r["agreement_score"]
        s["total_rounds"] += r["rounds"]
        if r["status"] in ("consensus", "partial_consensus"):
            s["agreed"] += 1

    return {
        "total_runs": total,
        "agreement_rate": agreement_rate,
        "avg_agreement_score": avg_score,
        "avg_rounds": avg_rounds,
        "early_termination_rate": early / total,
        "coordination_efficiency": coord_efficiency,
        "status_breakdown": {
            s: sum(1 for r in results if r["status"] == s)
            for s in ["consensus", "partial_consensus", "impasse", "error", "unknown"]
            if any(r["status"] == s for r in results)
        },
        "by_scenario": {
            name: {
                "agreement_rate": d["agreed"] / d["runs"],
                "avg_score": d["total_score"] / d["runs"],
                "avg_rounds": d["total_rounds"] / d["runs"],
            }
            for name, d in by_scenario.items()
        },
    }


def load_tool_metrics() -> dict:
    """Aggregate tool call stats from all *_events.jsonl files in output/."""
    output_dir = os.path.join(SCRIPT_DIR, "output")
    jsonl_files = sorted(glob.glob(os.path.join(output_dir, "*_events.jsonl")))
    if not jsonl_files:
        return {"total_tool_calls": 0, "successful_tool_calls": 0,
                "tool_success_rate": 0.0, "by_tool": {}}

    total_calls = 0
    successful_calls = 0
    by_tool: dict[str, dict] = defaultdict(lambda: {"total": 0, "success": 0})
    sessions_with_tools = 0
    sessions_total = 0
    retry_hints = 0
    retry_successes = 0

    for path in jsonl_files:
        session_had_tool = False
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "session_start":
                    sessions_total += 1
                elif etype == "tool_call":
                    tool = event.get("tool", "unknown")
                    success = bool(event.get("success", False))
                    total_calls += 1
                    by_tool[tool]["total"] += 1
                    if success:
                        successful_calls += 1
                        by_tool[tool]["success"] += 1
                    session_had_tool = True
                elif etype == "retry_hint":
                    retry_hints += 1
                elif etype == "retry_success":
                    retry_successes += 1
        if session_had_tool:
            sessions_with_tools += 1

    return {
        "total_tool_calls": total_calls,
        "successful_tool_calls": successful_calls,
        "tool_success_rate": successful_calls / total_calls if total_calls else 0.0,
        "sessions_total": sessions_total,
        "sessions_with_tool_use": sessions_with_tools,
        "retry_hints": retry_hints,
        "retry_successes": retry_successes,
        "retry_conversion_rate": retry_successes / retry_hints if retry_hints else 0.0,
        "by_tool": {
            tool: {
                "total": d["total"],
                "success": d["success"],
                "rate": d["success"] / d["total"] if d["total"] else 0.0,
            }
            for tool, d in sorted(by_tool.items())
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v:.1%}"

def _bar(v: float, width: int = 20) -> str:
    filled = round(v * width)
    return "█" * filled + "░" * (width - filled)


def print_report(results: list[dict], sm: dict, tm: dict) -> None:
    W = 64
    print("\n" + "═" * W)
    print("  EVALUATION REPORT  —  Multi-Agent Negotiation System")
    print("═" * W)

    if sm:
        print("\n  AGREEMENT METRICS")
        print("  " + "─" * 40)
        ar = sm["agreement_rate"]
        target = 0.89
        print(f"  Agreement rate      {_pct(ar):>8}  {_bar(ar)}  (target ≥89%)")
        print(f"  Avg agreement score {sm['avg_agreement_score']:>8.2f}  (consensus=1.0, partial=0.5, impasse=0.0)")
        print(f"  Total runs          {sm['total_runs']:>8}")

        print("\n  STATUS BREAKDOWN")
        print("  " + "─" * 40)
        for status, count in sm["status_breakdown"].items():
            print(f"  {status:<22} {count:>3}  {_bar(count / sm['total_runs'])}")

        print("\n  COORDINATION EFFICIENCY")
        print("  " + "─" * 40)
        print(f"  Avg rounds taken    {sm['avg_rounds']:>8.1f}")
        print(f"  Early terminations  {_pct(sm['early_termination_rate']):>8}  (rounds saved via declare_outcome)")
        print(f"  Coord efficiency    {_pct(sm['coordination_efficiency']):>8}  (rounds saved / max_rounds)")

        if sm.get("by_scenario"):
            print("\n  PER-SCENARIO BREAKDOWN")
            print("  " + "─" * 40)
            print(f"  {'Scenario':<22} {'Agree%':>8}  {'AvgScore':>9}  {'AvgRnds':>8}")
            for name, d in sm["by_scenario"].items():
                print(f"  {name:<22} {_pct(d['agreement_rate']):>8}  {d['avg_score']:>9.2f}  {d['avg_rounds']:>8.1f}")

    if tm["total_tool_calls"] > 0:
        print("\n  TOOL CALL METRICS  (ReAct agents)")
        print("  " + "─" * 40)
        print(f"  Overall success rate {_pct(tm['tool_success_rate']):>7}  "
              f"({tm['successful_tool_calls']}/{tm['total_tool_calls']} calls)")
        print(f"  Sessions with tools  {tm['sessions_with_tool_use']:>7} / {tm['sessions_total']}")
        if tm.get("retry_hints", 0) > 0:
            print(f"  Retry hints injected {tm['retry_hints']:>7}")
            print(f"  Retry successes      {tm['retry_successes']:>7}  "
                  f"({_pct(tm['retry_conversion_rate'])} of hints led to success)")
        print()
        print(f"  {'Tool':<28} {'Rate':>6}  {'Success':>7}  {'Total':>6}")
        for tool, d in tm["by_tool"].items():
            print(f"  {tool:<28} {_pct(d['rate']):>6}  {d['success']:>7}  {d['total']:>6}")
    else:
        print("\n  No tool call data found in output/ (SERPAPI_KEY may not be set).")

    if results:
        print("\n  RUN LOG")
        print("  " + "─" * 40)
        for r in results:
            early = " ✓early" if r["early_termination"] else ""
            print(f"  {r['scenario']:<22}  run {r['run']}  {r['status']:<20} "
                  f"{r['rounds']}/{r['max_rounds']} rnds{early}")

    print("\n" + "═" * W)

    # Resume-ready summary line
    if sm:
        print(f"\n  RESUME METRICS:")
        ar = sm["agreement_rate"]
        ce = sm["coordination_efficiency"]
        print(f"  • {_pct(ar)} agreement rate across {sm['total_runs']} runs / {len(sm.get('by_scenario', {}))} scenarios")
        if tm["total_tool_calls"] > 0:
            print(f"  • {_pct(tm['tool_success_rate'])} ReAct tool success rate ({tm['total_tool_calls']} calls)")
        if tm.get("retry_hints", 0) > 0:
            print(f"  • {_pct(tm['retry_conversion_rate'])} retry conversion rate "
                  f"({tm['retry_successes']}/{tm['retry_hints']} hints → success)")
        print(f"  • {_pct(ce)} avg coordination efficiency (early termination rate: {_pct(sm['early_termination_rate'])})")
        print()


def save_report(sm: dict, tm: dict, results: list[dict]) -> str:
    output_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(output_dir, f"eval_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"session_metrics": sm, "tool_metrics": tm, "runs": results}, f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate multi-agent negotiation across all scenarios"
    )
    parser.add_argument("--runs", type=int, default=2,
                        help="Number of runs per scenario (default: 2)")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(SCENARIOS.keys()),
                        help="Run a single scenario instead of all")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip running sessions — compute metrics from existing output files only")
    args = parser.parse_args()

    if args.report_only:
        results = []
        print("[INFO] --report-only: reading existing output files")
    else:
        results = run_all(scenario_filter=args.scenario, runs_per_scenario=args.runs)

    sm = compute_session_metrics(results)
    tm = load_tool_metrics()
    print_report(results, sm, tm)

    if sm or tm["total_tool_calls"] > 0:
        path = save_report(sm, tm, results)
        print(f"[REPORT] Saved to {path}")
