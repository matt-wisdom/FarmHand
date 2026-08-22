#!/usr/bin/env python3
"""
FarmHand Evaluation & Benchmarking Engine (evals/run_evals.py)
--------------------------------------------------------------
Runs automated benchmarks across Veterinary, Feed Formulation,
Agronomy, Tool Calling, Multi-Turn, and Language Fidelity test suites.

Usage:
  python evals/run_evals.py
  python evals/run_evals.py --category veterinary_diagnosis
  python evals/run_evals.py --language english
  python evals/run_evals.py --id eval_vet_01_goat_tetanus_en
  python evals/run_evals.py --save-report
"""

import os
import sys
import json
import time
import argparse
import re
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ROOT_DIR))

try:
    from backend.llm_engine import chat_completion
except ImportError:
    from llm_engine import chat_completion


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_dataset(dataset_path: Path) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_response(response_text: str, eval_spec: dict) -> tuple[bool, list[str]]:
    """
    Evaluates model response against the specification assertions.
    Returns (is_passed, list_of_failure_reasons).
    """
    failures = []

    # 1. Non-empty check
    if not response_text or not response_text.strip():
        failures.append("Response is empty.")
        return False, failures

    # 2. No raw JSON leakage
    raw_json_indicators = ['{"function_name"', '```json', '```JSON', '[{"ALIAS"']
    for rj in raw_json_indicators:
        if rj in response_text:
            failures.append(f"Raw JSON indicator found in user-facing response: {rj}")

    # 3. Positive assertions (must_contain)
    must_contain = eval_spec.get("must_contain", [])
    for term in must_contain:
        if not re.search(re.escape(term), response_text, re.IGNORECASE):
            failures.append(f"Missing required keyword/concept: '{term}'")

    # 4. Negative assertions (must_not_contain)
    must_not_contain = eval_spec.get("must_not_contain", [])
    for term in must_not_contain:
        if re.search(r"\b" + re.escape(term.strip()) + r"\b", response_text, re.IGNORECASE):
            failures.append(f"Prohibited term/particle detected: '{term}'")

    # 5. Format check (no forced list when prose required)
    if eval_spec.get("no_forced_list", False) or eval_spec.get("prose_required", False):
        lines = [ln.strip() for ln in response_text.splitlines() if ln.strip()]
        if len(lines) > 0 and (lines[0].startswith("1.") or (len(lines) > 1 and lines[1].startswith("1."))):
            if eval_spec.get("no_forced_list", False):
                failures.append("Response formatted as a forced numbered list instead of prose paragraph.")

    passed = len(failures) == 0
    return passed, failures


def run_benchmark(
    dataset_path: Path,
    filter_id: str | None = None,
    filter_cat: str | None = None,
    filter_lang: str | None = None,
    max_evals: int | None = None,
    save_report: bool = True,
    verbose: bool = False
):
    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN}            FarmHand AI Model Benchmark & Evaluation Suite             {RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    print(f"Dataset:  {dataset_path}")
    print(f"Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{CYAN}------------------------------------------------------------------------{RESET}\n")

    dataset = load_dataset(dataset_path)
    all_evals = dataset.get("evals", [])

    # Filter evals
    active_evals = []
    for ev in all_evals:
        if filter_id and ev.get("id") != filter_id:
            continue
        if filter_cat and ev.get("category") != filter_cat:
            continue
        if filter_lang and ev.get("language") != filter_lang:
            continue
        active_evals.append(ev)

    if max_evals:
        active_evals = active_evals[:max_evals]

    if not active_evals:
        print(f"{YELLOW}No evaluations match the specified filters.{RESET}")
        return

    print(f"Executing {len(active_evals)} benchmarks...\n")

    results = []
    start_total_time = time.time()

    for idx, ev in enumerate(active_evals, 1):
        eval_id = ev.get("id", f"eval_{idx}")
        category = ev.get("category", "general")
        lang = ev.get("language", "english")
        desc = ev.get("description", "")
        farm_id = ev.get("farm_id", "farm_9eb3f441")
        messages = ev.get("messages", [])

        print(f"[{idx}/{len(active_evals)}] {BOLD}{eval_id}{RESET} ({category} | {lang})")
        print(f"  Description: {desc}")
        if messages:
            last_msg = messages[-1].get("content", "")
            print(f"  Input: \"{last_msg[:90]}...\"" if len(last_msg) > 90 else f"  Input: \"{last_msg}\"")

        eval_start = time.time()
        try:
            response = chat_completion(
                messages=messages,
                farm_id=farm_id,
                language=lang
            )
            eval_latency = round(time.time() - eval_start, 2)
            passed, failure_reasons = evaluate_response(response, ev)
        except Exception as e:
            eval_latency = round(time.time() - eval_start, 2)
            response = f"ERROR: {str(e)}"
            passed = False
            failure_reasons = [f"Exception during execution: {str(e)}"]

        status_str = f"{GREEN}{BOLD}PASS{RESET}" if passed else f"{RED}{BOLD}FAIL{RESET}"
        print(f"  Result: {status_str} (Latency: {eval_latency}s)")
        if not passed:
            for r in failure_reasons:
                print(f"    - {RED}{r}{RESET}")
        if verbose or not passed:
            preview = response.replace("\n", " ")[:160]
            print(f"  Output: {preview}...")
        print()

        results.append({
            "id": eval_id,
            "category": category,
            "language": lang,
            "description": desc,
            "passed": passed,
            "latency_sec": eval_latency,
            "failure_reasons": failure_reasons,
            "response": response
        })

    total_duration = round(time.time() - start_total_time, 2)
    total_count = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = total_count - passed_count
    pass_rate = round((passed_count / total_count) * 100, 1) if total_count > 0 else 0.0
    avg_latency = round(sum(r["latency_sec"] for r in results) / total_count, 2) if total_count > 0 else 0.0

    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN}                         BENCHMARK SUMMARY                              {RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    print(f"Total Evaluations: {total_count}")
    print(f"Passed:            {GREEN}{passed_count}{RESET}")
    print(f"Failed:            {RED if failed_count > 0 else GREEN}{failed_count}{RESET}")
    print(f"Pass Rate:         {GREEN if pass_rate >= 90 else YELLOW}{pass_rate}%{RESET}")
    print(f"Average Latency:   {avg_latency}s / query")
    print(f"Total Duration:    {total_duration}s")
    print(f"{CYAN}------------------------------------------------------------------------{RESET}")

    cat_stats = {}
    for r in results:
        c = r["category"]
        if c not in cat_stats:
            cat_stats[c] = {"total": 0, "passed": 0}
        cat_stats[c]["total"] += 1
        if r["passed"]:
            cat_stats[c]["passed"] += 1

    print(f"{BOLD}Breakdown by Category:{RESET}")
    for c, s in cat_stats.items():
        rate = round((s["passed"] / s["total"]) * 100, 1)
        print(f"  - {c:<24}: {s['passed']}/{s['total']} passed ({rate}%)")
    print(f"{CYAN}========================================================================{RESET}\n")

    if save_report:
        reports_dir = ROOT_DIR / "evals" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_report_path = reports_dir / f"benchmark_{timestamp_str}.json"
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_count": total_count,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "pass_rate_pct": pass_rate,
                "avg_latency_sec": avg_latency,
                "total_duration_sec": total_duration,
                "category_stats": cat_stats,
                "results": results
            }, f, indent=2)

        md_report_path = reports_dir / f"benchmark_{timestamp_str}.md"
        md_lines = [
            f"# FarmHand AI Benchmark Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
            "",
            "## Summary",
            f"- **Total Tests**: {total_count}",
            f"- **Passed**: {passed_count}",
            f"- **Failed**: {failed_count}",
            f"- **Pass Rate**: {pass_rate}%",
            f"- **Average Latency**: {avg_latency}s",
            f"- **Total Duration**: {total_duration}s",
            "",
            "## Category Breakdown",
            "| Category | Passed | Total | Pass Rate |",
            "| :--- | :--- | :--- | :--- |"
        ]
        for c, s in cat_stats.items():
            rate = round((s["passed"] / s["total"]) * 100, 1)
            md_lines.append(f"| `{c}` | {s['passed']} | {s['total']} | {rate}% |")

        md_lines.extend([
            "",
            "## Detailed Results",
            "| ID | Category | Language | Status | Latency | Details |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |"
        ])
        for r in results:
            st = "✅ PASS" if r["passed"] else "❌ FAIL"
            details = ", ".join(r["failure_reasons"]) if not r["passed"] else "All assertions met"
            md_lines.append(f"| `{r['id']}` | `{r['category']}` | {r['language']} | {st} | {r['latency_sec']}s | {details} |")

        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        print(f"{GREEN}✓ Benchmark reports generated:{RESET}")
        print(f"  - Markdown: {md_report_path}")
        print(f"  - JSON:     {json_report_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FarmHand Evaluation Runner")
    parser.add_argument("--dataset", type=str, default=str(ROOT_DIR / "evals" / "eval_dataset.json"), help="Path to eval dataset JSON")
    parser.add_argument("--id", type=str, default=None, help="Run specific evaluation by ID")
    parser.add_argument("--category", type=str, default=None, help="Filter evaluations by category")
    parser.add_argument("--language", type=str, default=None, choices=["english", "pidgin"], help="Filter evaluations by language")
    parser.add_argument("--max-evals", type=int, default=None, help="Maximum number of evals to run")
    parser.add_argument("--save-report", action="store_true", default=True, help="Save JSON and Markdown benchmark reports")
    parser.add_argument("--no-report", action="store_false", dest="save_report", help="Disable report file generation")
    parser.add_argument("--verbose", action="store_true", help="Print verbose model outputs")

    args = parser.parse_args()
    run_benchmark(
        dataset_path=Path(args.dataset),
        filter_id=args.id,
        filter_cat=args.category,
        filter_lang=args.language,
        max_evals=args.max_evals,
        save_report=args.save_report,
        verbose=args.verbose
    )
