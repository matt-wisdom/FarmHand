# FarmHand AI Evaluation Suite

A comprehensive benchmarking framework for evaluating the FarmHand RAG + LLM pipeline.

## Quick Start

```bash
# Run all evaluations
python evals/run_evals.py

# Run only RAG evaluations
python evals/run_evals.py --eval=rag

# Run only LLM evaluations  
python evals/run_evals.py --eval=llm

# Run a specific evaluation
python evals/run_evals.py --eval=rag_returns_results

# Save results to JSON
python evals/run_evals.py --save=results.json

# Verbose output
python evals/run_evals.py --verbose
```

## Adding New Evaluations

Create a new eval function in `examples.py`:

```python
from evals.run_evals import register_eval, EvalResult
import time


@register_eval("my_new_eval")
def eval_my_new_test() -> EvalResult:
    start = time.time()
    try:
        # Your test code here
        passed = True
        details = "Test passed"
        score = 1.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)}"
        score = 0.0

    latency = (time.time() - start) * 1000
    return EvalResult("my_new_eval", passed, score, details, latency)
```

## Current Evaluations (15+)

### RAG Evaluations (14)
- `rag_returns_results` - Basic RAG returns results
- `rag_relevance_poultry` - Relevant results for poultry queries
- `rag_relevance_crops` - Relevant results for crop queries
- `rag_relevance_fish` - Relevant results for fish/aquaculture
- `rag_relevance_goat` - Relevant results for goat/sheep
- `rag_latency_acceptable` - Search completes in <3s
- `rag_bm25_working` - BM25 keyword search works
- `rag_faiss_working` - FAISS vector search works
- `rag_hybrid_combines_results` - Hybrid search combines sources
- `rag_result_has_scores` - Results include relevance scores
- `rag_result_has_metadata` - Results include filename/text
- `rag_handles_empty_query` - Handles minimal queries
- `rag_handles_special_chars` - Handles special characters
- `rag_diverse_results` - Returns diverse sources

### LLM Evaluations (3)
- `llm_responds_to_query` - LLM responds to queries
- `llm_json_output` - Returns valid JSON when requested
- `llm_uses_rag_context` - Incorporates RAG context

## Output

The framework outputs:
- Pass/Fail status for each eval
- Details about what was tested
- Latency in milliseconds
- Summary with overall score

Results can be saved to JSON for integration with CI/CD.