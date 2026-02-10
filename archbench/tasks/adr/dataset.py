"""
ADR (Architecture Decision Record) dataset loading.

Handles loading ADR datasets from CSV files or other formats.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import urllib.request
import shutil

from archbench.constants import (
    KEY_INSTANCE_ID,
    KEY_CONTEXT,
    KEY_DECISION,
    ADRInstance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# GitHub raw URL for the ADR dataset
GITHUB_RAW_URL = "https://raw.githubusercontent.com/sa4s-serc/ArchAI_ADR/main/data/0_shot.csv"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "archbench" / "adr"


def load_from_csv(
    csv_path: str,
    add_instance_ids: bool = True,
) -> List[ADRInstance]:
    """
    Load ADR dataset from the original ArchAI_ADR CSV format.

    Args:
        csv_path: Path to 0_shot.csv or few_shot.csv
        add_instance_ids: Whether to add instance IDs

    Returns:
        List of ADR instances with keys: instance_id, context, decision
    """
    instances = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            instance = {
                KEY_INSTANCE_ID: f"adr_{i:04d}" if add_instance_ids else row.get(KEY_INSTANCE_ID),
                KEY_CONTEXT: row["context"],
                KEY_DECISION: row["decision"],  # Ground truth
            }
            instances.append(instance)

    logger.info(f"Loaded {len(instances)} ADR instances from {csv_path}")
    return instances


def download_from_github(
    output_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Download ADR dataset directly from GitHub.

    Args:
        output_dir: Directory to save the dataset (default: ~/.cache/archbench/adr/)
        force: Force re-download even if file exists

    Returns:
        Path to downloaded CSV file
    """
    if output_dir is None:
        output_dir = DEFAULT_CACHE_DIR
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "0_shot.csv"

    if output_file.exists() and not force:
        logger.info(f"Using cached dataset: {output_file}")
        return str(output_file)

    logger.info(f"Downloading ADR dataset from GitHub...")
    logger.info(f"URL: {GITHUB_RAW_URL}")

    try:
        with urllib.request.urlopen(GITHUB_RAW_URL) as response:
            with open(output_file, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

        logger.info(f"Downloaded to: {output_file}")
        return str(output_file)
    except Exception as e:
        logger.error(f"Failed to download from GitHub: {e}")
        raise


def load_dataset(
    dataset_path: Optional[str] = None,
    instance_ids: Optional[List[str]] = None,
    download_if_missing: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load ADR dataset from file, GitHub, or HuggingFace.

    Args:
        dataset_path: Path to local dataset file (CSV/JSON/JSONL). If None, downloads from GitHub.
        instance_ids: Optional filter for specific instances
        download_if_missing: If True and dataset_path is None, download from GitHub

    Returns:
        List of dataset instances
    """
    if dataset_path is None:
        if download_if_missing:
            # Download from GitHub
            logger.info("No dataset path provided, downloading from GitHub...")
            dataset_path = download_from_github()
        else:
            # Try HuggingFace
            try:
                from datasets import load_dataset as hf_load_dataset
                hf_dataset = hf_load_dataset("archbench/adr", split="test")
                dataset = [dict(instance) for instance in hf_dataset]
                logger.info(f"Loaded {len(dataset)} ADR instances from HuggingFace")
            except Exception as e:
                logger.warning(f"Could not load from HuggingFace: {e}")
                raise ValueError("Please provide a local dataset_path or enable download_if_missing")

    if dataset_path:
        # Load from local file
        path = Path(dataset_path)
        if path.suffix == ".csv":
            dataset = load_from_csv(str(path))
        elif path.suffix == ".json":
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    dataset = list(data.values())
                else:
                    dataset = data
        elif path.suffix == ".jsonl":
            import json
            with open(path, "r", encoding="utf-8") as f:
                dataset = [json.loads(line) for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    # Filter by instance IDs if provided
    if instance_ids:
        instance_id_set = set(instance_ids)
        dataset = [d for d in dataset if d[KEY_INSTANCE_ID] in instance_id_set]

        # Check for missing IDs
        found_ids = {d[KEY_INSTANCE_ID] for d in dataset}
        missing_ids = instance_id_set - found_ids
        if missing_ids:
            logger.warning(f"Missing {len(missing_ids)} instance IDs: {missing_ids}")

    return dataset


def extract_prediction(raw_output: str) -> str:
    """
    Extract the ADR decision from the model response.
    Handles various output formats.
    """
    if not raw_output:
        return ""

    text = raw_output.strip()

    # Remove common prefixes
    prefixes_to_remove = [
        "## Decision",
        "**Decision:**",
        "Decision:",
        "### Decision",
    ]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Remove markdown formatting if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return text.strip()
