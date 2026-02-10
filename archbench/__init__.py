"""
ArchBench - A benchmark for evaluating LLMs on software architecture tasks.

This package provides:
- Standardized datasets for architecture tasks (ADR generation, traceability, etc.)
- Evaluation harness with metrics computation
- Inference pipeline with trajectory logging
- Submission validation and leaderboard integration

Usage:
    # Evaluation
    python -m archbench.harness.run_evaluation \
        --task adr \
        --predictions_path predictions.jsonl \
        --output_dir results/

    # Inference (generate predictions)
    python -m archbench.inference.run_inference \
        --task adr \
        --model gpt-4 \
        --output_path predictions.jsonl
"""

__version__ = "0.1.0"

from archbench.constants import (
    TASKS,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    KEY_RAW_OUTPUT,
)

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
    "__version__",
    "TASKS",
    "KEY_INSTANCE_ID",
    "KEY_MODEL",
    "KEY_PREDICTION",
    "KEY_RAW_OUTPUT",
    "load_dataset",
    "load_predictions",
    "validate_predictions",
    "compute_metrics",
    "compute_adr_metrics",
    "run_evaluation",
    "make_report",
]
