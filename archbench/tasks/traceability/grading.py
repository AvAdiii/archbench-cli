"""
Traceability Link Recovery (TLR) evaluation metrics.

Implements precision, recall, and F1 score for traceability links.
"""

import logging
from typing import List, Dict, Tuple, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def normalize_link(link: Tuple) -> Tuple:
    """
    Normalize a traceability link for exact comparison.

    Args:
        link: Tuple of (sentence_id, target)

    Returns:
        Normalized tuple (sentence_id, normalized_target)
    """
    sentence_id, target = link
    # Normalize: strip whitespace, normalize path separators
    target_normalized = str(target).strip().replace('\\', '/').rstrip('/')
    return (int(sentence_id), target_normalized)


def compute_traceability_metrics(
    predicted_links: List[Tuple],
    reference_links: List[Tuple],
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for traceability links using EXACT matching.

    Args:
        predicted_links: List of (sentence_id, target) tuples from model
        reference_links: List of (sentence_id, target) tuples from goldstandard

    Returns:
        Dictionary with precision, recall, f1, tp, fp, fn counts
    """
    if not reference_links:
        logger.warning("No reference links provided")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    # Normalize and convert to sets for exact matching
    pred_set = {normalize_link(link) for link in predicted_links}
    ref_set = {normalize_link(link) for link in reference_links}

    # Exact set-based matching
    true_positives = len(pred_set & ref_set)
    false_positives = len(pred_set - ref_set)
    false_negatives = len(ref_set - pred_set)

    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def compute_traceability_metrics_batch(
    predictions: List[List[Tuple]],
    references: List[List[Tuple]],
) -> Dict[str, List[float]]:
    """
    Compute metrics for a batch of predictions.

    Args:
        predictions: List of predicted link lists
        references: List of reference link lists

    Returns:
        Dictionary mapping metric names to lists of scores
    """
    n = len(predictions)
    assert len(references) == n, "Predictions and references must have same length"

    results = {
        "precision": [],
        "recall": [],
        "f1": [],
        "true_positives": [],
        "false_positives": [],
        "false_negatives": [],
    }

    for pred_links, ref_links in zip(predictions, references):
        metrics = compute_traceability_metrics(pred_links, ref_links)
        for key in results:
            results[key].append(metrics[key])

    return results
