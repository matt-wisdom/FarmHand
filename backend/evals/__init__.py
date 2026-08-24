"""
FarmHand AI Evaluation Suite
=============================

Add new evaluations to examples.py or create new modules.
Run with: python evals/run_evals.py
"""

from .run_evals import EVAL_REGISTRY, EvalResult, run_all_evals, run_evaluation

__all__ = ["EVAL_REGISTRY", "EvalResult", "run_all_evals", "run_evaluation"]
