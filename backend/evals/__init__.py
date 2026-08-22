"""
FarmHand AI Evaluation Suite
=============================

Add new evaluations to examples.py or create new modules.
Run with: python evals/run_evals.py
"""
from .run_evals import run_all_evals, run_evaluation, EvalResult, EVAL_REGISTRY

__all__ = ["run_all_evals", "run_evaluation", "EvalResult", "EVAL_REGISTRY"]