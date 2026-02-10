"""
ArchBench Evaluation Harness.

This module provides the evaluation pipeline for ArchBench tasks.
"""

from archbench.harness.utils import (
    load_dataset,
    load_predictions,
    validate_predictions,
)

from archbench.harness.grading import (
    compute_metrics,
    compute_adr_metrics,
)

from archbench.harness.run_evaluation import (
    run_evaluation,
    make_report,
)

__all__ = [
    "load_dataset",
    "load_predictions",
    "validate_predictions",
    "compute_metrics",
    "compute_adr_metrics",
    "run_evaluation",
    "make_report",
]
