"""
Utility functions for ArchBench evaluation harness.

Handles dataset loading, prediction loading, and validation.
Utility functions for dataset loading, prediction I/O, and validation.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from archbench.constants import (
    TASKS,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    KEY_RAW_OUTPUT,
    KEY_CONTEXT,
    KEY_DECISION,
    ADRInstance,
    Prediction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset(
    task: str,
    split: str = "test",
    dataset_path: Optional[str] = None,
    instance_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load ArchBench dataset for a specific task.

    Args:
        task: Task name (adr, serverless, dynamic, traceability)
        split: Dataset split (train, dev, test)
        dataset_path: Path to local dataset file (JSON/JSONL/CSV), or None to load from HuggingFace
        instance_ids: Optional list of specific instance IDs to load

    Returns:
        List of dataset instances

    Example:
        >>> dataset = load_dataset("adr", split="test")
        >>> print(len(dataset))
        95
        >>> print(dataset[0].keys())
        dict_keys(['instance_id', 'context', 'decision'])
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}. Available tasks: {list(TASKS.keys())}")

    if dataset_path is not None:
        # Load from local file
        dataset = load_dataset_from_file(dataset_path)
    else:
        # Try to load from HuggingFace
        try:
            from datasets import load_dataset as hf_load_dataset
            hf_dataset = hf_load_dataset(TASKS[task]["dataset"], split=split)
            dataset = [dict(instance) for instance in hf_dataset]
        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            logger.warning("Please provide a local dataset_path")
            raise

    # Filter by instance IDs if provided
    if instance_ids:
        instance_id_set = set(instance_ids)
        dataset = [d for d in dataset if d[KEY_INSTANCE_ID] in instance_id_set]

        # Check for missing IDs
        found_ids = {d[KEY_INSTANCE_ID] for d in dataset}
        missing_ids = instance_id_set - found_ids
        if missing_ids:
            logger.warning(f"Missing {len(missing_ids)} instance IDs from dataset: {missing_ids}")

    logger.info(f"Loaded {len(dataset)} instances for task '{task}'")
    return dataset


def load_dataset_from_file(path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from a local file (JSON, JSONL, or CSV).

    Args:
        path: Path to the dataset file

    Returns:
        List of dataset instances
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # Could be {instance_id: instance} format
                data = list(data.values())
            return data

    elif path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    elif path.suffix == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
            # Add instance_id if not present
            for i, row in enumerate(data):
                if KEY_INSTANCE_ID not in row:
                    row[KEY_INSTANCE_ID] = f"instance_{i:04d}"
            return data
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json, .jsonl, or .csv")


def load_adr_dataset_from_csv(
    csv_path: str,
    add_instance_ids: bool = True,
) -> List[ADRInstance]:
    """
    Load ADR dataset from the original ArchAI_ADR CSV format.

    Args:
        csv_path: Path to 0_shot.csv or few_shot.csv
        add_instance_ids: Whether to add instance IDs

    Returns:
        List of ADR instances
    """
    instances = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            instance = {
                KEY_INSTANCE_ID: f"adr_{i:04d}" if add_instance_ids else row.get(KEY_INSTANCE_ID),
                KEY_CONTEXT: row["context"],
                KEY_DECISION: row["decision"],  # This is the ground truth
            }
            instances.append(instance)

    logger.info(f"Loaded {len(instances)} ADR instances from {csv_path}")
    return instances


# =============================================================================
# Prediction Loading
# =============================================================================

def load_predictions(
    predictions_path: str,
) -> Dict[str, Prediction]:
    """
    Load predictions from a file.

    Args:
        predictions_path: Path to predictions file (JSON or JSONL)

    Returns:
        Dictionary mapping instance_id to prediction

    Example:
        >>> preds = load_predictions("predictions.jsonl")
        >>> print(preds["adr_0001"]["prediction"][:50])
        "We decided to use PostgreSQL because..."
    """
    path = Path(predictions_path)

    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
            if isinstance(predictions, list):
                predictions = {p[KEY_INSTANCE_ID]: p for p in predictions}
    elif path.suffix == ".jsonl":
        predictions = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pred = json.loads(line)
                    predictions[pred[KEY_INSTANCE_ID]] = pred
    else:
        raise ValueError(f"Unsupported predictions format: {path.suffix}. Use .json or .jsonl")

    logger.info(f"Loaded {len(predictions)} predictions from {path}")
    return predictions


def get_predictions_from_file(
    predictions_path: str,
    dataset: Optional[List[Dict]] = None,
) -> List[Prediction]:
    """
    Load predictions and optionally validate against dataset.
    Load predictions from a JSONL file into a dictionary keyed by instance_id.

    Args:
        predictions_path: Path to predictions file
        dataset: Optional dataset to validate against

    Returns:
        List of predictions
    """
    predictions_dict = load_predictions(predictions_path)
    predictions = list(predictions_dict.values())

    # Validate each prediction has required fields
    for pred in predictions:
        if KEY_INSTANCE_ID not in pred:
            raise ValueError(f"Prediction missing required field: {KEY_INSTANCE_ID}")
        if KEY_PREDICTION not in pred:
            raise ValueError(f"Prediction {pred[KEY_INSTANCE_ID]} missing required field: {KEY_PREDICTION}")

    # Validate against dataset if provided
    if dataset:
        dataset_ids = {d[KEY_INSTANCE_ID] for d in dataset}
        pred_ids = {p[KEY_INSTANCE_ID] for p in predictions}

        missing_in_dataset = pred_ids - dataset_ids
        if missing_in_dataset:
            logger.warning(f"Predictions for unknown instances: {missing_in_dataset}")

        missing_predictions = dataset_ids - pred_ids
        if missing_predictions:
            logger.warning(f"Missing predictions for {len(missing_predictions)} instances")

    return predictions


# =============================================================================
# Prediction Validation
# =============================================================================

def validate_predictions(
    predictions: Union[List[Dict], Dict[str, Dict]],
    task: str,
) -> Dict[str, List[str]]:
    """
    Validate predictions for a specific task.

    Args:
        predictions: List of predictions or dict mapping instance_id to prediction
        task: Task name

    Returns:
        Dictionary with 'valid' and 'invalid' instance IDs
    """
    if isinstance(predictions, dict):
        predictions = list(predictions.values())

    valid_ids = []
    invalid_ids = []
    errors = []

    for pred in predictions:
        instance_id = pred.get(KEY_INSTANCE_ID, "unknown")

        # Check required fields
        if KEY_INSTANCE_ID not in pred:
            invalid_ids.append(instance_id)
            errors.append(f"{instance_id}: missing instance_id")
            continue

        if KEY_PREDICTION not in pred:
            invalid_ids.append(instance_id)
            errors.append(f"{instance_id}: missing prediction")
            continue

        # Check prediction is not empty
        prediction_value = pred[KEY_PREDICTION]
        if prediction_value is None:
            invalid_ids.append(instance_id)
            errors.append(f"{instance_id}: empty prediction")
            continue

        # Handle both string (ADR) and list (traceability) predictions
        if isinstance(prediction_value, str) and prediction_value.strip() == "":
            invalid_ids.append(instance_id)
            errors.append(f"{instance_id}: empty prediction")
            continue
        elif isinstance(prediction_value, list) and len(prediction_value) == 0:
            invalid_ids.append(instance_id)
            errors.append(f"{instance_id}: empty prediction")
            continue

        # Task-specific validation
        if task == "traceability":
            # Traceability should have a list of trace links
            try:
                links = pred[KEY_PREDICTION]
                if isinstance(links, str):
                    links = json.loads(links)
                if not isinstance(links, list):
                    invalid_ids.append(instance_id)
                    errors.append(f"{instance_id}: trace_links should be a list")
                    continue
            except json.JSONDecodeError:
                invalid_ids.append(instance_id)
                errors.append(f"{instance_id}: invalid JSON in prediction")
                continue

        valid_ids.append(instance_id)

    if errors:
        logger.warning(f"Validation errors:\n" + "\n".join(errors[:10]))
        if len(errors) > 10:
            logger.warning(f"... and {len(errors) - 10} more errors")

    logger.info(f"Validation: {len(valid_ids)} valid, {len(invalid_ids)} invalid")

    return {
        "valid": valid_ids,
        "invalid": invalid_ids,
        "errors": errors,
    }


# =============================================================================
# Output Writing
# =============================================================================

def save_predictions(
    predictions: List[Dict],
    output_path: str,
    format: str = "jsonl",
) -> None:
    """
    Save predictions to a file.

    Args:
        predictions: List of prediction dictionaries
        output_path: Path to output file
        format: Output format (jsonl or json)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    elif format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved {len(predictions)} predictions to {path}")


def save_report(
    report: Dict,
    output_path: str,
) -> None:
    """
    Save evaluation report to a JSON file.

    Args:
        report: Evaluation report dictionary
        output_path: Path to output file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved report to {path}")
