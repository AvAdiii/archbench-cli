"""
Traceability Link Recovery (TLR) dataset loading.

Handles loading ArDoCo benchmark for SAD-SAM and SAD-Code traceability.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import urllib.request
import zipfile
import shutil

from archbench.constants import KEY_INSTANCE_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Zenodo download URL (v1.1 - specific version)
ZENODO_RECORD_ID = "16743302"
ZENODO_DOWNLOAD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/ardoco/benchmark-v1.1.zip"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "archbench" / "traceability"

# Available projects in the benchmark
PROJECTS = ["bigbluebutton", "jabref", "mediastore", "teammates", "teastore"]


def download_from_zenodo(
    output_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Download ArDoCo benchmark from Zenodo.

    Args:
        output_dir: Directory to save the benchmark (default: ~/.cache/archbench/traceability/)
        force: Force re-download even if already exists

    Returns:
        Path to extracted benchmark directory
    """
    if output_dir is None:
        output_dir = DEFAULT_CACHE_DIR
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = output_dir / "benchmark"

    if benchmark_dir.exists() and not force:
        logger.info(f"Using cached benchmark: {benchmark_dir}")
        return str(benchmark_dir)

    zip_file = output_dir / "benchmark.zip"

    logger.info(f"Downloading ArDoCo benchmark from Zenodo...")
    logger.info(f"URL: {ZENODO_DOWNLOAD_URL}")

    try:
        with urllib.request.urlopen(ZENODO_DOWNLOAD_URL) as response:
            with open(zip_file, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

        logger.info(f"Extracting benchmark...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # The zip extracts to ardoco-benchmark-<commit_hash>/ directory
        # Find it and rename to just "benchmark" for consistency
        ardoco_dirs = list(output_dir.glob("ardoco-benchmark-*"))
        if ardoco_dirs and not benchmark_dir.exists():
            ardoco_dirs[0].rename(benchmark_dir)

        # Clean up zip file
        zip_file.unlink()

        if not benchmark_dir.exists():
            raise FileNotFoundError(f"Benchmark directory not found after extraction: {benchmark_dir}")

        logger.info(f"Benchmark extracted to: {benchmark_dir}")
        return str(benchmark_dir)
    except Exception as e:
        logger.error(f"Failed to download from Zenodo: {e}")
        raise


def load_text(text_file: Path) -> List[str]:
    """
    Load SAD text file and return list of sentences.

    Args:
        text_file: Path to text file

    Returns:
        List of sentences (indexed from 1)
    """
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Filter out empty lines and strip whitespace
    sentences = [""]  # Index 0 is empty (sentences start at 1)
    for line in lines:
        line = line.strip()
        if line:
            sentences.append(line)

    return sentences


def load_goldstandard_sad_sam(csv_file: Path) -> List[Tuple[str, int]]:
    """
    Load goldstandard for SAD-SAM traceability.

    Args:
        csv_file: Path to goldstandard CSV

    Returns:
        List of (model_element_id, sentence_number) tuples
    """
    links = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row['modelElementID']
            sentence_num = int(row['sentence'])
            links.append((model_id, sentence_num))

    return links


def load_goldstandard_sad_code(csv_file: Path) -> List[Tuple[int, str]]:
    """
    Load goldstandard for SAD-Code traceability.

    Args:
        csv_file: Path to goldstandard CSV

    Returns:
        List of (sentence_number, code_path) tuples
    """
    links = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence_num = int(row['sentenceID'])
            code_path = row['codeID']
            links.append((sentence_num, code_path))

    return links


def load_project(
    project_name: str,
    benchmark_dir: str,
    task_type: str = "sad-code",
) -> Dict[str, Any]:
    """
    Load a single project from the benchmark.

    Args:
        project_name: Name of project (e.g., 'mediastore')
        benchmark_dir: Path to benchmark root directory
        task_type: Type of traceability task ('sad-sam' or 'sad-code')

    Returns:
        Dictionary with project data
    """
    project_dir = Path(benchmark_dir) / project_name

    if not project_dir.exists():
        raise FileNotFoundError(f"Project not found: {project_dir}")

    # Find text directories and goldstandards
    text_dirs = sorted(project_dir.glob("text_*"), reverse=True)  # Try newest first
    if not text_dirs:
        raise FileNotFoundError(f"No text directory found in {project_dir}")

    goldstandards_dir = project_dir / "goldstandards"

    # Try to find a text directory with matching goldstandard
    text_dir = None
    year = None
    goldstandard_file = None

    for candidate_text_dir in text_dirs:
        candidate_year = candidate_text_dir.name.split("_")[1]

        if task_type == "sad-sam":
            # Try to find matching SAD-SAM goldstandard
            candidate_goldstandard = goldstandards_dir / f"goldstandard_sad_{candidate_year}-sam_{candidate_year}.csv"
            if candidate_goldstandard.exists():
                text_dir = candidate_text_dir
                year = candidate_year
                goldstandard_file = candidate_goldstandard
                break
        elif task_type == "sad-code":
            # Try to find matching SAD-Code goldstandard
            # First try exact match
            candidate_goldstandard = goldstandards_dir / f"goldstandard_sad_{candidate_year}-code_{candidate_year}.csv"
            if candidate_goldstandard.exists():
                text_dir = candidate_text_dir
                year = candidate_year
                goldstandard_file = candidate_goldstandard
                break
            # Then try any goldstandard with this SAD year
            matches = list(goldstandards_dir.glob(f"goldstandard_sad_{candidate_year}-code_*.csv"))
            if matches:
                text_dir = candidate_text_dir
                year = candidate_year
                goldstandard_file = matches[0]
                break

    if text_dir is None or goldstandard_file is None:
        raise FileNotFoundError(f"No {task_type} goldstandard found for {project_name}")

    # Load text file (usually projectname.txt)
    text_files = list(text_dir.glob("*.txt"))
    if not text_files:
        raise FileNotFoundError(f"No text file found in {text_dir}")

    sentences = load_text(text_files[0])

    # Load goldstandard
    if task_type == "sad-sam":
        goldstandard = load_goldstandard_sad_sam(goldstandard_file)
        # Extract unique model elements
        available_targets = sorted(set(target for _, target in goldstandard))
    elif task_type == "sad-code":
        goldstandard = load_goldstandard_sad_code(goldstandard_file)
        # Extract unique code files (these are the actual paths the LLM should choose from)
        available_targets = sorted(set(target for _, target in goldstandard))
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return {
        KEY_INSTANCE_ID: f"{project_name}_{year}",
        "project": project_name,
        "year": year,
        "sentences": sentences,
        "goldstandard": goldstandard,
        "available_targets": available_targets,  # NEW: Actual code files to choose from
        "num_sentences": len(sentences) - 1,  # -1 because index 0 is empty
        "num_links": len(goldstandard),
    }


def load_dataset(
    dataset_path: Optional[str] = None,
    instance_ids: Optional[List[str]] = None,
    projects: Optional[List[str]] = None,
    task_type: str = "sad-code",
    download_if_missing: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load traceability dataset from ArDoCo benchmark.

    Args:
        dataset_path: Path to benchmark directory. If None, downloads from Zenodo.
        instance_ids: Optional filter for specific instances
        projects: Optional filter for specific projects (e.g., ['mediastore', 'teastore'])
        task_type: Type of traceability ('sad-sam' or 'sad-code')
        download_if_missing: If True and dataset_path is None, download from Zenodo

    Returns:
        List of dataset instances
    """
    if dataset_path is None:
        if download_if_missing:
            logger.info("No dataset path provided, downloading from Zenodo...")
            dataset_path = download_from_zenodo()
        else:
            raise ValueError("Please provide dataset_path or enable download_if_missing")

    benchmark_dir = Path(dataset_path)
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    # Determine which projects to load
    if projects is None:
        projects = PROJECTS

    # Load all projects
    dataset = []
    for project_name in projects:
        try:
            project_data = load_project(project_name, str(benchmark_dir), task_type)
            dataset.append(project_data)
            logger.info(f"Loaded {project_name}: {project_data['num_sentences']} sentences, "
                       f"{project_data['num_links']} links")
        except FileNotFoundError as e:
            logger.warning(f"Skipping {project_name}: {e}")
            continue

    # Filter by instance IDs if provided
    if instance_ids:
        instance_id_set = set(instance_ids)
        dataset = [d for d in dataset if d[KEY_INSTANCE_ID] in instance_id_set]

    logger.info(f"Loaded {len(dataset)} projects for {task_type} traceability")
    return dataset


def extract_prediction(raw_output: str, task_type: str = "sad-code") -> List[Tuple]:
    """
    Extract traceability links from model response.

    Expected format: JSON array of links
    [
      {"sentence": 1, "target": "path/to/file.java"},
      {"sentence": 3, "target": "Component_123"}
    ]

    Args:
        raw_output: Raw text from LLM
        task_type: Type of traceability ('sad-sam' or 'sad-code')

    Returns:
        List of (sentence_id, target) tuples
    """
    if not raw_output:
        return []

    import json
    import re

    # Try to find JSON in the response
    json_match = re.search(r'\[[\s\S]*\]', raw_output)
    if json_match:
        try:
            links_json = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from response")
            return []
    else:
        # Try parsing the whole response as JSON
        try:
            links_json = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.warning("Could not parse traceability response as JSON")
            return []

    # Convert to tuples
    links = []
    for link in links_json:
        if isinstance(link, dict):
            sentence_id = link.get('sentence') or link.get('sentence_id')
            target = link.get('target') or link.get('code') or link.get('model_element')

            if sentence_id is not None and target:
                try:
                    links.append((int(sentence_id), str(target)))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid link format: {link}")
                    continue

    return links
