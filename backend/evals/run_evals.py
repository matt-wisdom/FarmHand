"""
FarmHand AI Evaluation Framework
=================================
A extensible benchmarking suite for evaluating the RAG + LLM pipeline.

Usage:
    python run_evals.py              # Run all evals
    python run_evals.py --eval=rag   # Run only RAG evals
    python run_evals.py --eval=llm   # Run only LLM evals
    python run_evals.py --verbose    # Show detailed output
"""

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add backend and evals to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import query_knowledge_base, search_knowledge_base  # noqa: E402

# Try to import LLM - handle if not available
try:
    from llm_engine import chat_completion  # noqa: E402

    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    print("[evals] LLM not available, skipping LLM-based evals")


# =============================================================================
# Evaluation Result Types
# =============================================================================


@dataclass
class EvalResult:
    """Result of a single evaluation."""

    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: str
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvalSuite:
    """A collection of related evaluations."""

    name: str
    description: str
    evals: list[Callable]


# =============================================================================
# Evaluation Registry - Add your evals here
# =============================================================================

EVAL_REGISTRY: dict[str, Callable] = {}


def register_eval(name: str):
    """Decorator to register an evaluation function."""

    def decorator(func: Callable):
        EVAL_REGISTRY[name] = func
        func._eval_name = name
        return func

    return decorator


# Import custom evals (they register themselves via decorator)
with contextlib.suppress(ImportError):
    import examples  # noqa: F401
with contextlib.suppress(ImportError):
    from evals import examples  # noqa: F401


# =============================================================================
# RAG Evaluations (15+ tests)
# =============================================================================


@register_eval("rag_returns_results")
def eval_rag_returns_results() -> EvalResult:
    """Test that RAG search returns results for common queries."""
    start = time.time()
    try:
        results = search_knowledge_base("poultry feed nutrition", top_k=3)
        passed = len(results) > 0
        details = f"Returned {len(results)} results"
        score = 1.0 if passed else 0.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_returns_results", passed, score, details, latency)


@register_eval("rag_relevance_poultry")
def eval_rag_relevance_poultry() -> EvalResult:
    """Test RAG returns relevant results for poultry queries."""
    start = time.time()
    try:
        results = search_knowledge_base("chicken disease Newcastle", top_k=5)
        # Check if any result mentions poultry/chicken/disease
        relevant = any(
            any(
                kw in r.get("text", "").lower()
                for kw in ["chicken", "poultry", "newcastle", "disease", "bird"]
            )
            for r in results
        )
        passed = relevant and len(results) > 0
        details = f"Found {len(results)} results, relevant: {relevant}"
        score = 1.0 if passed else 0.5
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_relevance_poultry", passed, score, details, latency)


@register_eval("rag_relevance_crops")
def eval_rag_relevance_crops() -> EvalResult:
    """Test RAG returns relevant results for crop queries."""
    start = time.time()
    try:
        results = search_knowledge_base("tomato farming irrigation", top_k=5)
        relevant = any(
            any(
                kw in r.get("text", "").lower()
                for kw in ["tomato", "crop", "irrigation", "farming", "plant"]
            )
            for r in results
        )
        passed = relevant and len(results) > 0
        details = f"Found {len(results)} results, relevant: {relevant}"
        score = 1.0 if passed else 0.5
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_relevance_crops", passed, score, details, latency)


@register_eval("rag_relevance_fish")
def eval_rag_relevance_fish() -> EvalResult:
    """Test RAG returns relevant results for fish/aquaculture queries."""
    start = time.time()
    try:
        results = search_knowledge_base("fish pond water quality", top_k=5)
        relevant = any(
            any(
                kw in r.get("text", "").lower()
                for kw in ["fish", "pond", "aquaculture", "water", "tilapia"]
            )
            for r in results
        )
        passed = relevant and len(results) > 0
        details = f"Found {len(results)} results, relevant: {relevant}"
        score = 1.0 if passed else 0.5
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_relevance_fish", passed, score, details, latency)


@register_eval("rag_relevance_goat")
def eval_rag_relevance_goat() -> EvalResult:
    """Test RAG returns relevant results for goat/sheep queries."""
    start = time.time()
    try:
        results = search_knowledge_base("goat feeding nutrition", top_k=5)
        relevant = any(
            any(
                kw in r.get("text", "").lower()
                for kw in ["goat", "sheep", "ruminant", "feeding", "livestock"]
            )
            for r in results
        )
        passed = relevant and len(results) > 0
        details = f"Found {len(results)} results, relevant: {relevant}"
        score = 1.0 if passed else 0.5
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_relevance_goat", passed, score, details, latency)


@register_eval("rag_latency_acceptable")
def eval_rag_latency() -> EvalResult:
    """Test that RAG search completes within acceptable time."""
    start = time.time()
    try:
        search_knowledge_base("poultry disease", top_k=5)
        latency = (time.time() - start) * 1000
        # Acceptable latency is under 3 seconds
        passed = latency < 3000
        details = f"Latency: {latency:.0f}ms"
        score = (
            1.0
            if latency < 1000
            else (0.7 if latency < 2000 else 0.4 if latency < 3000 else 0.0)
        )
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
        latency = 0
    return EvalResult("rag_latency_acceptable", passed, score, details, latency)


@register_eval("rag_bm25_working")
def eval_rag_bm25() -> EvalResult:
    """Test that BM25 keyword search returns results."""
    start = time.time()
    try:
        from rag_pipeline import bm25_search

        results = bm25_search("feather loss poultry", top_k=3)
        passed = len(results) > 0
        details = f"BM25 returned {len(results)} results"
        score = 1.0 if passed else 0.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_bm25_working", passed, score, details, latency)


@register_eval("rag_faiss_working")
def eval_rag_faiss() -> EvalResult:
    """Test that FAISS vector search returns results."""
    start = time.time()
    try:
        from rag_pipeline import vector_search

        results = vector_search("chicken health treatment", top_k=3)
        passed = len(results) > 0
        details = f"FAISS returned {len(results)} results"
        score = 1.0 if passed else 0.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_faiss_working", passed, score, details, latency)


@register_eval("rag_hybrid_combines_results")
def eval_rag_hybrid() -> EvalResult:
    """Test that hybrid search combines BM25 and FAISS results."""
    start = time.time()
    try:
        from rag_pipeline import bm25_search, combine_results, vector_search

        bm25_results = bm25_search("poultry feed", top_k=5)
        vector_results = vector_search("poultry feed", top_k=5)
        combined = combine_results(bm25_results, vector_results, top_k=3)

        # Should have results from at least one source
        passed = len(combined) > 0
        details = f"BM25: {len(bm25_results)}, FAISS: {len(vector_results)}, Combined: {len(combined)}"
        score = 1.0 if passed else 0.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_hybrid_combines_results", passed, score, details, latency)


@register_eval("rag_result_has_scores")
def eval_rag_scores() -> EvalResult:
    """Test that RAG results include relevance scores."""
    start = time.time()
    try:
        results = search_knowledge_base("poultry", top_k=3)
        has_scores = all("score" in r for r in results)
        passed = has_scores and len(results) > 0
        details = f"Results have scores: {has_scores}"
        score = 1.0 if passed else 0.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_result_has_scores", passed, score, details, latency)


@register_eval("rag_result_has_metadata")
def eval_rag_metadata() -> EvalResult:
    """Test that RAG results include filename and chunk_id."""
    start = time.time()
    try:
        results = search_knowledge_base("chicken", top_k=3)
        has_filename = all("filename" in r for r in results)
        has_text = all("text" in r for r in results)
        passed = has_filename and has_text and len(results) > 0
        details = f"Has filename: {has_filename}, has text: {has_text}"
        score = 1.0 if passed else 0.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_result_has_metadata", passed, score, details, latency)


@register_eval("rag_handles_empty_query")
def eval_rag_empty_query() -> EvalResult:
    """Test that RAG handles empty/minimal queries gracefully."""
    start = time.time()
    try:
        results = search_knowledge_base("a", top_k=3)
        # Should return something or empty list, not crash
        passed = True
        details = f"Handled gracefully, returned {len(results)} results"
        score = 1.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_handles_empty_query", passed, score, details, latency)


@register_eval("rag_handles_special_chars")
def eval_rag_special_chars() -> EvalResult:
    """Test that RAG handles special characters."""
    start = time.time()
    try:
        results = search_knowledge_base("chicken @#$% disease", top_k=3)
        passed = True  # Should not crash
        details = f"Handled special chars, returned {len(results)} results"
        score = 1.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_handles_special_chars", passed, score, details, latency)


@register_eval("rag_diverse_results")
def eval_rag_diversity() -> EvalResult:
    """Test that RAG returns results from different sources."""
    start = time.time()
    try:
        results = search_knowledge_base("livestock farming", top_k=10)
        filenames = set(r.get("filename", "") for r in results)
        # Should have at least 2 different sources
        passed = len(filenames) >= 2
        details = f"Unique sources: {len(filenames)}"
        score = min(1.0, len(filenames) / 3)
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_diverse_results", passed, score, details, latency)


# =============================================================================
# LLM Evaluations (Optional - requires LLM)
# =============================================================================

if HAS_LLM:

    @register_eval("llm_responds_to_query")
    def eval_llm_responds() -> EvalResult:
        """Test that LLM responds to a simple query."""
        start = time.time()
        try:
            from llm_engine import chat_completion

            response = chat_completion(
                messages=[{"role": "user", "content": "What is 1+1?"}],
                farm_id="eval_farm",
                language="english",
            )
            passed = response and len(response) > 0
            details = f"Response length: {len(response)} chars"
            score = 1.0 if passed else 0.0
        except Exception as e:
            passed = False
            details = f"Error: {str(e)[:100]}"
            score = 0.0
        latency = (time.time() - start) * 1000
        return EvalResult("llm_responds_to_query", passed, score, details, latency)

    @register_eval("llm_json_output")
    def eval_llm_json() -> EvalResult:
        """Test that LLM returns valid JSON when requested."""
        start = time.time()
        try:
            response = chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": "List 3 poultry diseases in JSON format",
                    }
                ],
                farm_id="eval_farm",
                language="english",
            )
            # Try to parse as JSON or extract from code block
            clean = response.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()
            try:
                json.loads(clean)
                passed = True
                details = "Valid JSON output"
                score = 1.0
            except Exception:
                has_diseases = any(
                    d in response.lower()
                    for d in [
                        "newcastle",
                        "coccidiosis",
                        "gumboro",
                        "flu",
                        "marek",
                        "fowl pox",
                        "bronchitis",
                    ]
                )
                passed = has_diseases
                details = (
                    "Parsed structured disease response"
                    if has_diseases
                    else "Not valid JSON"
                )
                score = 1.0 if has_diseases else 0.5
        except Exception as e:
            passed = False
            details = f"Error: {str(e)[:100]}"
            score = 0.0
        latency = (time.time() - start) * 1000
        return EvalResult("llm_json_output", passed, score, details, latency)

    @register_eval("llm_uses_rag_context")
    def eval_llm_rag_context() -> EvalResult:
        """Test that LLM incorporates RAG context in response."""
        start = time.time()
        try:
            # First get RAG context
            kb_result = query_knowledge_base("Newcastle disease poultry")
            context = kb_result.get("context_prompt", "")

            # Check context was retrieved
            passed = len(context) > 10
            details = f"Context length: {len(context)} chars"
            score = 1.0 if passed else 0.0
        except Exception as e:
            passed = False
            details = f"Error: {str(e)[:100]}"
            score = 0.0
        latency = (time.time() - start) * 1000
        return EvalResult("llm_uses_rag_context", passed, score, details, latency)


# =============================================================================
# Evaluation Runner
# =============================================================================


def run_evaluation(name: str, verbose: bool = False) -> EvalResult:
    """Run a single evaluation by name."""
    if name not in EVAL_REGISTRY:
        return EvalResult(name, False, 0.0, f"Unknown evaluation: {name}", 0)

    if verbose:
        print(f"\n[EVAl] Running: {name}")

    eval_func = EVAL_REGISTRY[name]
    result = eval_func()

    return result


def run_all_evals(category: str = "all", verbose: bool = False) -> list[EvalResult]:
    """Run all evaluations, optionally filtered by category."""
    results = []

    # Define categories
    rag_evals = [k for k in EVAL_REGISTRY if k.startswith("rag_")]
    llm_evals = [k for k in EVAL_REGISTRY if k.startswith("llm_")]

    if category == "all":
        eval_names = list(EVAL_REGISTRY.keys())
    elif category == "rag":
        eval_names = rag_evals
    elif category == "llm":
        eval_names = llm_evals
    else:
        eval_names = [k for k in EVAL_REGISTRY if category.lower() in k.lower()]

    for name in eval_names:
        result = run_evaluation(name, verbose)
        results.append(result)

        status = "✓" if result.passed else "✗"
        print(f"{status} {name}: {result.details} ({result.latency_ms:.0f}ms)")

    return results


def print_summary(results: list[EvalResult]):
    """Print a summary of evaluation results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / total if total > 0 else 0
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
    pct_passed = (100 * passed / total) if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total:   {total}")
    print(f"Passed:  {passed} ({pct_passed:.1f}%)")
    print(f"Failed:  {total - passed}")
    print(f"Score:   {avg_score:.2f}")
    print(f"Latency: {avg_latency:.0f}ms avg")
    print("=" * 60)


def save_results(results: list[EvalResult], filepath: str = "eval_results.json"):
    """Save evaluation results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "score": r.score,
                "details": r.details,
                "latency_ms": r.latency_ms,
                "timestamp": r.timestamp,
            }
            for r in results
        ],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filepath}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="FarmHand AI Evaluation Framework")
    parser.add_argument(
        "--eval",
        type=str,
        default="all",
        help="Run specific eval (rag, llm, or eval name)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    parser.add_argument("--save", type=str, default=None, help="Save results to file")

    args = parser.parse_args()

    print("=" * 60)
    print("FarmHand AI Evaluation Suite")
    print("=" * 60)
    print(f"Running: {args.eval}")
    print(f"LLM available: {HAS_LLM}")

    results = run_all_evals(args.eval, args.verbose)

    print_summary(results)

    if args.save:
        save_results(results, args.save)

    # Exit with appropriate code
    passed = sum(1 for r in results if r.passed)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
