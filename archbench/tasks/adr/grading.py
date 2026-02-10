"""
ADR (Architecture Decision Record) evaluation metrics.

Implements text similarity metrics:
- ROUGE-1, ROUGE-2, ROUGE-L
- BLEU
- METEOR
- BERTScore (optional, requires torch)
"""

import logging
from typing import Dict, List
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Lazy Loading of Metrics
# =============================================================================

_METRICS_CACHE = {}


def _get_rouge():
    if "rouge" not in _METRICS_CACHE:
        try:
            from evaluate import load
            _METRICS_CACHE["rouge"] = load("rouge")
        except ImportError:
            logger.warning("evaluate not installed. Install with: pip install archbench[eval]")
            _METRICS_CACHE["rouge"] = None
    return _METRICS_CACHE["rouge"]


def _get_bleu():
    if "bleu" not in _METRICS_CACHE:
        try:
            from evaluate import load
            _METRICS_CACHE["bleu"] = load("bleu")
        except ImportError:
            logger.warning("evaluate not installed. Install with: pip install archbench[eval]")
            _METRICS_CACHE["bleu"] = None
    return _METRICS_CACHE["bleu"]


def _get_meteor():
    if "meteor" not in _METRICS_CACHE:
        try:
            from evaluate import load
            _METRICS_CACHE["meteor"] = load("meteor")
        except ImportError:
            logger.warning("evaluate not installed. Install with: pip install archbench[eval]")
            _METRICS_CACHE["meteor"] = None
    return _METRICS_CACHE["meteor"]


def _get_bertscore():
    if "bertscore" not in _METRICS_CACHE:
        try:
            from evaluate import load
            _METRICS_CACHE["bertscore"] = load("bertscore")
        except ImportError:
            logger.warning("BERTScore not available. Install with: pip install archbench[bertscore]")
            _METRICS_CACHE["bertscore"] = None
    return _METRICS_CACHE["bertscore"]


def _bertscore_available() -> bool:
    """Check if BERTScore is available (torch installed)."""
    try:
        import torch
        return True
    except ImportError:
        return False


# =============================================================================
# ADR Metrics
# =============================================================================

def compute_adr_metrics(
    prediction: str,
    reference: str,
    compute_bertscore: bool = True,
) -> Dict[str, float]:
    """
    Compute all metrics for a single ADR prediction.

    Args:
        prediction: Generated ADR decision text
        reference: Ground truth ADR decision text
        compute_bertscore: Whether to compute BERTScore (slower but more accurate)

    Returns:
        Dictionary of metric scores
    """
    if not prediction or not reference:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bleu": 0.0,
            "meteor": 0.0,
            "bertscore_p": 0.0,
            "bertscore_r": 0.0,
            "bertscore_f1": 0.0,
        }

    metrics = {}

    # ROUGE scores
    rouge = _get_rouge()
    if rouge is not None:
        try:
            rouge_results = rouge.compute(
                predictions=[prediction],
                references=[reference]
            )
            metrics["rouge1"] = rouge_results["rouge1"]
            metrics["rouge2"] = rouge_results["rouge2"]
            metrics["rougeL"] = rouge_results["rougeL"]
        except Exception as e:
            logger.warning(f"ROUGE computation failed: {e}")
            metrics["rouge1"] = 0.0
            metrics["rouge2"] = 0.0
            metrics["rougeL"] = 0.0
    else:
        metrics["rouge1"] = None
        metrics["rouge2"] = None
        metrics["rougeL"] = None

    # BLEU score
    bleu = _get_bleu()
    if bleu is not None:
        try:
            bleu_results = bleu.compute(
                predictions=[prediction],
                references=[[reference]]  # BLEU expects list of reference lists
            )
            metrics["bleu"] = bleu_results["bleu"]
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            metrics["bleu"] = 0.0
    else:
        metrics["bleu"] = None

    # METEOR score
    meteor = _get_meteor()
    if meteor is not None:
        try:
            meteor_results = meteor.compute(
                predictions=[prediction],
                references=[reference]
            )
            metrics["meteor"] = meteor_results["meteor"]
        except Exception as e:
            logger.warning(f"METEOR computation failed: {e}")
            metrics["meteor"] = 0.0
    else:
        metrics["meteor"] = None

    # BERTScore (optional, requires torch)
    if compute_bertscore and _bertscore_available():
        try:
            bertscore = _get_bertscore()
            if bertscore is not None:
                bert_results = bertscore.compute(
                    predictions=[prediction],
                    references=[reference],
                    lang="en"
                )
                metrics["bertscore_p"] = float(np.mean(bert_results["precision"]))
                metrics["bertscore_r"] = float(np.mean(bert_results["recall"]))
                metrics["bertscore_f1"] = float(np.mean(bert_results["f1"]))
            else:
                metrics["bertscore_p"] = None
                metrics["bertscore_r"] = None
                metrics["bertscore_f1"] = None
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            metrics["bertscore_p"] = 0.0
            metrics["bertscore_r"] = 0.0
            metrics["bertscore_f1"] = 0.0
    else:
        if compute_bertscore and not _bertscore_available():
            logger.info("BERTScore skipped (torch not installed). Install with: pip install archbench[bertscore]")
        metrics["bertscore_p"] = None
        metrics["bertscore_r"] = None
        metrics["bertscore_f1"] = None

    return metrics


def compute_adr_metrics_batch(
    predictions: List[str],
    references: List[str],
    compute_bertscore: bool = True,
) -> Dict[str, List[float]]:
    """
    Compute metrics for a batch of ADR predictions (more efficient).

    Args:
        predictions: List of generated ADR decision texts
        references: List of ground truth ADR decision texts
        compute_bertscore: Whether to compute BERTScore

    Returns:
        Dictionary mapping metric names to lists of scores
    """
    n = len(predictions)
    assert len(references) == n, "Predictions and references must have same length"

    # Filter out empty predictions
    valid_pairs = [
        (i, p, r) for i, (p, r) in enumerate(zip(predictions, references))
        if p and r and p.strip() and r.strip()
    ]

    if not valid_pairs:
        return {
            "rouge1": [0.0] * n,
            "rouge2": [0.0] * n,
            "rougeL": [0.0] * n,
            "bleu": [0.0] * n,
            "meteor": [0.0] * n,
            "bertscore_p": [0.0] * n,
            "bertscore_r": [0.0] * n,
            "bertscore_f1": [0.0] * n,
        }

    valid_indices, valid_preds, valid_refs = zip(*valid_pairs)

    # Initialize results
    results = {
        "rouge1": [0.0] * n,
        "rouge2": [0.0] * n,
        "rougeL": [0.0] * n,
        "bleu": [0.0] * n,
        "meteor": [0.0] * n,
        "bertscore_p": [0.0] * n,
        "bertscore_r": [0.0] * n,
        "bertscore_f1": [0.0] * n,
    }

    # Compute ROUGE in batch
    try:
        rouge = _get_rouge()
        for idx, pred, ref in zip(valid_indices, valid_preds, valid_refs):
            rouge_result = rouge.compute(predictions=[pred], references=[ref])
            results["rouge1"][idx] = rouge_result["rouge1"]
            results["rouge2"][idx] = rouge_result["rouge2"]
            results["rougeL"][idx] = rouge_result["rougeL"]
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")

    # Compute BLEU
    try:
        bleu = _get_bleu()
        for idx, pred, ref in zip(valid_indices, valid_preds, valid_refs):
            bleu_result = bleu.compute(predictions=[pred], references=[[ref]])
            results["bleu"][idx] = bleu_result["bleu"]
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")

    # Compute METEOR
    try:
        meteor = _get_meteor()
        for idx, pred, ref in zip(valid_indices, valid_preds, valid_refs):
            meteor_result = meteor.compute(predictions=[pred], references=[ref])
            results["meteor"][idx] = meteor_result["meteor"]
    except Exception as e:
        logger.warning(f"METEOR computation failed: {e}")

    # Compute BERTScore in batch (more efficient, requires torch)
    if compute_bertscore and _bertscore_available():
        try:
            bertscore = _get_bertscore()
            if bertscore is not None:
                bert_results = bertscore.compute(
                    predictions=list(valid_preds),
                    references=list(valid_refs),
                    lang="en"
                )
                for i, idx in enumerate(valid_indices):
                    results["bertscore_p"][idx] = float(bert_results["precision"][i])
                    results["bertscore_r"][idx] = float(bert_results["recall"][i])
                    results["bertscore_f1"][idx] = float(bert_results["f1"][i])
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
    elif compute_bertscore:
        logger.info("BERTScore skipped (torch not installed). Install with: pip install archbench[bertscore]")

    return results
