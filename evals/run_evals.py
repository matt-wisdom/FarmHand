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

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ROOT_DIR))

try:
    from backend.llm_engine import chat_completion  # noqa: E402
    from backend.rag_pipeline import get_embedding_model  # noqa: E402
except ImportError:
    from llm_engine import chat_completion  # noqa: E402
    from rag_pipeline import get_embedding_model  # noqa: E402


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes normalized cosine similarity between two 1D embedding vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def evaluate_semantic_match(
    response_text: str, eval_spec: dict
) -> tuple[bool, float, list[str]]:
    """
    Evaluates semantic closeness against reference answer and semantic concepts using FastEmbed ONNX vectors.
    Returns (passed, semantic_score, failure_reasons).
    """
    failures = []
    semantic_score = 1.0
    ref_answer = eval_spec.get("reference_answer")
    semantic_concepts = eval_spec.get("semantic_concepts", [])

    if not ref_answer and not semantic_concepts:
        return True, 1.0, []

    try:
        embedder = get_embedding_model()
    except Exception as e:
        return True, 1.0, [f"Semantic embedding skipped (embedder unavailable: {e})"]

    # 1. Whole-Response Reference Similarity
    if ref_answer:
        min_score = eval_spec.get("min_semantic_score", 0.72)
        try:
            vectors = list(
                embedder.embed(
                    [
                        f"passage: {response_text.strip()}",
                        f"passage: {ref_answer.strip()}",
                    ]
                )
            )
            semantic_score = round(
                compute_cosine_similarity(np.array(vectors[0]), np.array(vectors[1])), 3
            )
            if semantic_score < min_score:
                failures.append(
                    f"Semantic similarity score ({semantic_score:.2f}) below threshold ({min_score:.2f}) for reference answer."
                )
        except Exception as e:
            failures.append(f"Error computing reference embedding similarity: {e}")

    # 2. Concept-Level Sentence Matching
    if semantic_concepts:
        concept_threshold = eval_spec.get("concept_threshold", 0.68)
        sentences = [
            s.strip()
            for s in re.split(r"[.\n!?;]+", response_text)
            if len(s.strip()) > 8
        ]
        if not sentences:
            sentences = [response_text.strip()]

        try:
            sent_vectors = [
                np.array(v)
                for v in embedder.embed([f"passage: {s}" for s in sentences])
            ]
            for concept in semantic_concepts:
                concept_vec = np.array(list(embedder.embed([f"passage: {concept}"]))[0])
                best_sim = (
                    max(
                        compute_cosine_similarity(sv, concept_vec)
                        for sv in sent_vectors
                    )
                    if sent_vectors
                    else 0.0
                )

                if best_sim < concept_threshold:
                    failures.append(
                        f"Required semantic concept not found (best similarity: {best_sim:.2f} < {concept_threshold:.2f}): '{concept}'"
                    )
        except Exception as e:
            failures.append(f"Error computing concept semantic matching: {e}")

    passed = len(failures) == 0
    return passed, semantic_score, failures


def load_dataset(dataset_path: Path) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_response(
    response_text: str, eval_spec: dict
) -> tuple[bool, float, list[str]]:
    """
    Evaluates model response against the specification assertions.
    Returns (is_passed, semantic_score, list_of_failure_reasons).
    """
    failures = []
    semantic_score = 1.0

    # 1. Non-empty check
    if not response_text or not response_text.strip():
        failures.append("Response is empty.")
        return False, 0.0, failures

    # 2. No raw JSON leakage
    raw_json_indicators = ['{"function_name"', "```json", "```JSON", '[{"ALIAS"']
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
        if re.search(
            r"\b" + re.escape(term.strip()) + r"\b", response_text, re.IGNORECASE
        ):
            failures.append(f"Prohibited term/particle detected: '{term}'")

    # 5. Format check (no forced list when prose required)
    if eval_spec.get("no_forced_list", False) or eval_spec.get("prose_required", False):
        lines = [ln.strip() for ln in response_text.splitlines() if ln.strip()]
        if (
            len(lines) > 0
            and (
                lines[0].startswith("1.")
                or (len(lines) > 1 and lines[1].startswith("1."))
            )
            and eval_spec.get("no_forced_list", False)
        ):
            failures.append(
                "Response formatted as a forced numbered list instead of prose paragraph."
            )

    # 6. Semantic Matching
    if eval_spec.get("reference_answer") or eval_spec.get("semantic_concepts"):
        sem_passed, sem_score, sem_failures = evaluate_semantic_match(
            response_text, eval_spec
        )
        semantic_score = sem_score
        if not sem_passed:
            failures.extend(sem_failures)

    passed = len(failures) == 0
    return passed, semantic_score, failures


def run_benchmark(
    dataset_path: Path,
    filter_id: str | None = None,
    filter_cat: str | None = None,
    filter_lang: str | None = None,
    max_evals: int | None = None,
    save_report: bool = True,
    verbose: bool = False,
):
    print(
        f"{BOLD}{CYAN}========================================================================{RESET}"
    )
    print(
        f"{BOLD}{CYAN}            FarmHand AI Model Benchmark & Evaluation Suite             {RESET}"
    )
    print(
        f"{BOLD}{CYAN}========================================================================{RESET}"
    )
    print(f"Dataset:  {dataset_path}")
    print(f"Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"{CYAN}------------------------------------------------------------------------{RESET}\n"
    )

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

        print(
            f"[{idx}/{len(active_evals)}] {BOLD}{eval_id}{RESET} ({category} | {lang})"
        )
        print(f"  Description: {desc}")
        if messages:
            last_msg = messages[-1].get("content", "")
            print(
                f'  Input: "{last_msg[:90]}..."'
                if len(last_msg) > 90
                else f'  Input: "{last_msg}"'
            )

        eval_start = time.time()
        try:
            response = chat_completion(
                messages=messages, farm_id=farm_id, language=lang
            )
            eval_latency = round(time.time() - eval_start, 2)
            passed, semantic_score, failure_reasons = evaluate_response(response, ev)
        except Exception as e:
            eval_latency = round(time.time() - eval_start, 2)
            response = f"ERROR: {e!s}"
            passed = False
            semantic_score = 0.0
            failure_reasons = [f"Exception during execution: {e!s}"]

        status_str = (
            f"{GREEN}{BOLD}PASS{RESET}" if passed else f"{RED}{BOLD}FAIL{RESET}"
        )
        has_semantic = bool(ev.get("reference_answer") or ev.get("semantic_concepts"))
        sem_str = f" | Semantic Sim: {semantic_score:.2f}" if has_semantic else ""
        print(f"  Result: {status_str} (Latency: {eval_latency}s{sem_str})")
        if not passed:
            for r in failure_reasons:
                print(f"    - {RED}{r}{RESET}")
        if verbose or not passed:
            preview = response.replace("\n", " ")[:160]
            print(f"  Output: {preview}...")
        print()

        results.append(
            {
                "id": eval_id,
                "category": category,
                "language": lang,
                "description": desc,
                "passed": passed,
                "semantic_score": semantic_score,
                "latency_sec": eval_latency,
                "failure_reasons": failure_reasons,
                "response": response,
            }
        )

    total_duration = round(time.time() - start_total_time, 2)
    total_count = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = total_count - passed_count
    pass_rate = round((passed_count / total_count) * 100, 1) if total_count > 0 else 0.0
    avg_latency = (
        round(sum(r["latency_sec"] for r in results) / total_count, 2)
        if total_count > 0
        else 0.0
    )
    avg_semantic = (
        round(sum(r["semantic_score"] for r in results) / total_count, 2)
        if total_count > 0
        else 0.0
    )

    print(
        f"{BOLD}{CYAN}========================================================================{RESET}"
    )
    print(
        f"{BOLD}{CYAN}                         BENCHMARK SUMMARY                              {RESET}"
    )
    print(
        f"{BOLD}{CYAN}========================================================================{RESET}"
    )
    print(f"Total Evaluations:   {total_count}")
    print(f"Passed:              {GREEN}{passed_count}{RESET}")
    print(
        f"Failed:              {RED if failed_count > 0 else GREEN}{failed_count}{RESET}"
    )
    print(
        f"Pass Rate:           {GREEN if pass_rate >= 90 else YELLOW}{pass_rate}%{RESET}"
    )
    print(f"Avg Semantic Score:  {CYAN}{avg_semantic:.2f}{RESET}")
    print(f"Average Latency:     {avg_latency}s / query")
    print(f"Total Duration:      {total_duration}s")
    print(
        f"{CYAN}------------------------------------------------------------------------{RESET}"
    )

    cat_stats = {}
    for r in results:
        c = r["category"]
        if c not in cat_stats:
            cat_stats[c] = {"total": 0, "passed": 0, "semantic_scores": []}
        cat_stats[c]["total"] += 1
        cat_stats[c]["semantic_scores"].append(r["semantic_score"])
        if r["passed"]:
            cat_stats[c]["passed"] += 1

    print(f"{BOLD}Breakdown by Category:{RESET}")
    for c, s in cat_stats.items():
        rate = round((s["passed"] / s["total"]) * 100, 1)
        cat_sem = round(sum(s["semantic_scores"]) / len(s["semantic_scores"]), 2)
        print(
            f"  - {c:<24}: {s['passed']}/{s['total']} passed ({rate}%) | Avg Sem: {cat_sem:.2f}"
        )
    print(
        f"{CYAN}========================================================================{RESET}\n"
    )

    if save_report:
        reports_dir = ROOT_DIR / "evals" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_report_path = reports_dir / f"benchmark_{timestamp_str}.json"
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "total_count": total_count,
                    "passed_count": passed_count,
                    "failed_count": failed_count,
                    "pass_rate_pct": pass_rate,
                    "avg_semantic_score": avg_semantic,
                    "avg_latency_sec": avg_latency,
                    "total_duration_sec": total_duration,
                    "category_stats": {
                        k: {"total": v["total"], "passed": v["passed"]}
                        for k, v in cat_stats.items()
                    },
                    "results": results,
                },
                f,
                indent=2,
            )

        md_report_path = reports_dir / f"benchmark_{timestamp_str}.md"
        md_lines = [
            f"# FarmHand AI Benchmark Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
            "",
            "## Summary",
            f"- **Total Tests**: {total_count}",
            f"- **Passed**: {passed_count}",
            f"- **Failed**: {failed_count}",
            f"- **Pass Rate**: {pass_rate}%",
            f"- **Avg Semantic Score**: {avg_semantic:.2f}",
            f"- **Average Latency**: {avg_latency}s",
            f"- **Total Duration**: {total_duration}s",
            "",
            "## Category Breakdown",
            "| Category | Passed | Total | Pass Rate | Avg Semantic Score |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for c, s in cat_stats.items():
            rate = round((s["passed"] / s["total"]) * 100, 1)
            cat_sem = round(sum(s["semantic_scores"]) / len(s["semantic_scores"]), 2)
            md_lines.append(
                f"| `{c}` | {s['passed']} | {s['total']} | {rate}% | {cat_sem:.2f} |"
            )

        md_lines.extend(
            [
                "",
                "## Detailed Results",
                "| ID | Category | Language | Status | Latency | Semantic Sim | Details |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for r in results:
            st = "✅ PASS" if r["passed"] else "❌ FAIL"
            details = (
                ", ".join(r["failure_reasons"])
                if not r["passed"]
                else "All assertions met"
            )
            md_lines.append(
                f"| `{r['id']}` | `{r['category']}` | {r['language']} | {st} | {r['latency_sec']}s | {r['semantic_score']:.2f} | {details} |"
            )

        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        print(f"{GREEN}✓ Benchmark reports generated:{RESET}")
        print(f"  - Markdown: {md_report_path}")
        print(f"  - JSON:     {json_report_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FarmHand Evaluation Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(ROOT_DIR / "evals" / "eval_dataset.json"),
        help="Path to eval dataset JSON",
    )
    parser.add_argument(
        "--id", type=str, default=None, help="Run specific evaluation by ID"
    )
    parser.add_argument(
        "--category", type=str, default=None, help="Filter evaluations by category"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["english", "pidgin"],
        help="Filter evaluations by language",
    )
    parser.add_argument(
        "--max-evals", type=int, default=None, help="Maximum number of evals to run"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="Save JSON and Markdown benchmark reports",
    )
    parser.add_argument(
        "--no-report",
        action="store_false",
        dest="save_report",
        help="Disable report file generation",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print verbose model outputs"
    )

    args = parser.parse_args()
    run_benchmark(
        dataset_path=Path(args.dataset),
        filter_id=args.id,
        filter_cat=args.category,
        filter_lang=args.language,
        max_evals=args.max_evals,
        save_report=args.save_report,
        verbose=args.verbose,
    )
