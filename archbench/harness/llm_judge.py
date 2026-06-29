"""
LLM-as-a-Judge evaluation for ArchBench.

Builds a single prompt per task containing:
- Aggregated metrics from the evaluation report
- A few sampled (prediction, reference) pairs
Then makes ONE LLM call to get an overall qualitative assessment.

Usage:
    archbench judge --task adr --predictions_path preds.jsonl \
        --dataset_path data.csv --judge_model gpt-4
"""

import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from archbench.constants import (
    TASKS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    KEY_DECISION,
    KEY_CONTEXT,
)
from archbench.harness.utils import (
    load_predictions,
    save_report,
)

logger = logging.getLogger(__name__)

MAX_TEXT_LEN = 2000  # Truncate individual text fields
DEFAULT_SAMPLE_COUNT = 5  # How many instances to include in the prompt


def _truncate(text: str, max_len: int = MAX_TEXT_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... [truncated, {len(text)} chars total]"


def _format_metrics_summary(report: Dict) -> str:
    """Format aggregated metrics from a report.json into a readable summary."""
    metrics = report.get("metrics", {})
    if not metrics:
        return "(no automated metrics available)"

    lines = []
    for key, val in sorted(metrics.items()):
        if key.endswith("_mean"):
            name = key.replace("_mean", "")
            lines.append(f"  {name}: {val:.4f}")
    return "\n".join(lines) if lines else "(no metrics)"


def _format_instance_metrics(metrics: Dict) -> str:
    """Format per-instance metrics dict into one line."""
    if not metrics:
        return ""
    parts = [f"{k}: {v:.3f}" for k, v in metrics.items() if isinstance(v, (int, float))]
    return ", ".join(parts)


# =============================================================================
# Task-Specific Prompt Builders (each returns a single prompt for ALL samples)
# =============================================================================

def build_adr_overview_prompt(
    sampled_instances: List[Dict],
    predictions: Dict[str, Dict],
    instance_metrics: Dict[str, Dict],
    report: Optional[Dict] = None,
) -> List[Dict[str, str]]:
    """
    Build a single overview prompt for ADR judge.
    Includes aggregated metrics + sampled (context, reference, prediction) triples.
    """
    # Aggregated metrics section
    metrics_summary = "(not provided)"
    if report:
        metrics_summary = _format_metrics_summary(report)

    # Build sampled examples
    examples = []
    for i, inst in enumerate(sampled_instances, 1):
        iid = inst[KEY_INSTANCE_ID]
        context = _truncate(inst.get(KEY_CONTEXT, ""), 1000)
        reference = _truncate(inst.get(KEY_DECISION, ""), 1000)
        pred = predictions.get(iid, {})
        prediction = _truncate(str(pred.get(KEY_PREDICTION, "")), 1000)
        inst_metrics = _format_instance_metrics(instance_metrics.get(iid, {}))

        example = (
            f"### Instance {i} ({iid})\n"
            f"**Context:** {context}\n\n"
            f"**Reference Decision:** {reference}\n\n"
            f"**Generated Decision:** {prediction}\n"
        )
        if inst_metrics:
            example += f"**Metrics:** {inst_metrics}\n"
        examples.append(example)

    examples_text = "\n---\n".join(examples)

    system = (
        "You are an expert evaluator for software architecture benchmarks. "
        "You will review the overall performance of an LLM on an Architecture Decision Record (ADR) generation task.\n\n"
        "ADR task: Given architectural context, the model generates a Decision section. "
        "The decision should state what was decided and provide rationale.\n\n"
        "You will be given:\n"
        "1. Aggregated automated metrics across all evaluated instances\n"
        "2. A few sampled instances showing (context, reference, prediction)\n\n"
        "Provide an overall qualitative assessment in JSON:\n"
        "{\n"
        '  "score": <1-5>,\n'
        '  "summary": "<2-3 sentence overall assessment>",\n'
        '  "strengths": ["<strength 1>", ...],\n'
        '  "weaknesses": ["<weakness 1>", ...],\n'
        '  "examples": ["<cite a specific example from the instances, e.g. In instance X the model correctly/incorrectly ...>", ...]\n'
        "}\n\n"
        "IMPORTANT: In 'examples', reference specific instances by ID and quote short snippets "
        "from the generated vs reference decisions to illustrate your points.\n\n"
        "Score scale:\n"
        "  1 = Very poor overall quality\n"
        "  2 = Below average; significant issues\n"
        "  3 = Acceptable; reasonable but room for improvement\n"
        "  4 = Good; captures decisions well with minor gaps\n"
        "  5 = Excellent; consistently high quality"
    )

    user = (
        f"## Aggregated Metrics (across all evaluated instances)\n{metrics_summary}\n\n"
        f"## Sampled Instances ({len(sampled_instances)} of {report.get('completed_instances', '?')} total)\n\n"
        f"{examples_text}\n\n"
        "Provide your overall assessment as JSON."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_traceability_overview_prompt(
    sampled_instances: List[Dict],
    predictions: Dict[str, Dict],
    instance_metrics: Dict[str, Dict],
    report: Optional[Dict] = None,
) -> List[Dict[str, str]]:
    """
    Build a single overview prompt for traceability judge.
    Since link lists can be huge, we show metrics + a compact sample.
    """
    metrics_summary = "(not provided)"
    if report:
        metrics_summary = _format_metrics_summary(report)

    examples = []
    for i, inst in enumerate(sampled_instances, 1):
        iid = inst[KEY_INSTANCE_ID]
        project = inst.get("project", iid)
        ref_links = inst.get("goldstandard", [])
        pred = predictions.get(iid, {})
        pred_links = pred.get(KEY_PREDICTION, [])
        inst_metrics = _format_instance_metrics(instance_metrics.get(iid, {}))

        # Show counts + a small sample of links
        ref_sample = ref_links[:10] if len(ref_links) > 10 else ref_links
        pred_sample = pred_links[:10] if isinstance(pred_links, list) and len(pred_links) > 10 else pred_links

        example = (
            f"### Instance {i} ({project})\n"
            f"**Reference links:** {len(ref_links)} total, sample: {json.dumps(ref_sample, ensure_ascii=False)}\n"
            f"**Predicted links:** {len(pred_links) if isinstance(pred_links, list) else '?'} total, "
            f"sample: {json.dumps(pred_sample, ensure_ascii=False)}\n"
        )
        if inst_metrics:
            example += f"**Metrics:** {inst_metrics}\n"
        examples.append(example)

    examples_text = "\n---\n".join(examples)

    system = (
        "You are an expert evaluator for software architecture benchmarks. "
        "You will review the overall performance of an LLM on a traceability link recovery task.\n\n"
        "Traceability task: Given architecture documentation sentences, the model identifies "
        "which code files relate to each sentence (trace links).\n\n"
        "You will be given:\n"
        "1. Aggregated automated metrics (Precision, Recall, F1)\n"
        "2. A few sampled instances showing reference vs predicted links\n\n"
        "Provide an overall qualitative assessment in JSON:\n"
        "{\n"
        '  "score": <1-5>,\n'
        '  "summary": "<2-3 sentence overall assessment>",\n'
        '  "strengths": ["<strength 1>", ...],\n'
        '  "weaknesses": ["<weakness 1>", ...],\n'
        '  "examples": ["<cite a specific example from the instances, e.g. In instance X the model correctly/incorrectly ...>", ...]\n'
        "}\n\n"
        "IMPORTANT: In 'examples', reference specific instances and cite concrete link matches/misses.\n\n"
        "Score scale:\n"
        "  1 = Very poor; almost no correct links\n"
        "  2 = Below average; many missed or wrong links\n"
        "  3 = Acceptable; partial recovery\n"
        "  4 = Good; most important links recovered\n"
        "  5 = Excellent; comprehensive and accurate"
    )

    user = (
        f"## Aggregated Metrics\n{metrics_summary}\n\n"
        f"## Sampled Instances ({len(sampled_instances)} of {report.get('completed_instances', '?')} total)\n\n"
        f"{examples_text}\n\n"
        "Provide your overall assessment as JSON."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


JUDGE_PROMPT_BUILDERS = {
    "adr": build_adr_overview_prompt,
    "traceability": build_traceability_overview_prompt,
}


# =============================================================================
# Response Parsing
# =============================================================================

def parse_judge_response(response: str) -> Dict[str, Any]:
    """Parse the judge response JSON, handling markdown fences."""
    text = response.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    def _extract(d):
        return {
            "score": max(1, min(5, int(d.get("score", 0)))),
            "summary": d.get("summary", ""),
            "strengths": d.get("strengths", []),
            "weaknesses": d.get("weaknesses", []),
            "examples": d.get("examples", []),
        }

    try:
        return _extract(json.loads(text))
    except (json.JSONDecodeError, ValueError, TypeError):
        # Try to find JSON object in response
        match = re.search(r'\{.*"score".*\}', text, re.DOTALL)
        if match:
            try:
                return _extract(json.loads(match.group(0)))
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning(f"Could not parse judge response: {text[:200]}")
        return {"score": 0, "summary": f"Parse error: {text[:300]}", "strengths": [], "weaknesses": [], "examples": []}


# =============================================================================
# Main Judge Function
# =============================================================================

def run_llm_judge(
    task: str,
    predictions_path: str,
    dataset_path: Optional[str] = None,
    judge_model: str = "gpt-4",
    output_dir: str = "results",
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    report_path: Optional[str] = None,
    instance_metrics_path: Optional[str] = None,
    run_id: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    ollama_host: str = "http://localhost:11434",
) -> Dict[str, Any]:
    """
    Run LLM-as-a-judge: one prompt, one call, one overall assessment.

    Args:
        task: Task name (adr, traceability)
        predictions_path: Path to predictions JSONL
        dataset_path: Path to dataset (for loading references)
        judge_model: Model to use as judge
        output_dir: Where to save judge report
        sample_count: Number of instances to sample for the prompt
        report_path: Path to evaluation report.json (for aggregated metrics)
        instance_metrics_path: Path to per-instance metrics JSONL
        run_id: Unique run identifier
        max_tokens: Max tokens for judge response
        temperature: Temperature for judge

    Returns:
        Judge report dictionary
    """
    from archbench.inference.run_inference import get_provider

    start_time = time.time()

    if run_id is None:
        run_id = f"judge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if task not in JUDGE_PROMPT_BUILDERS:
        raise ValueError(
            f"LLM judge not supported for task '{task}'. "
            f"Available: {list(JUDGE_PROMPT_BUILDERS.keys())}"
        )

    build_prompt = JUDGE_PROMPT_BUILDERS[task]

    # Load dataset
    logger.info(f"Loading dataset for {task}...")
    if task == "adr":
        from archbench.tasks.adr import dataset as adr_dataset
        dataset = adr_dataset.load_dataset(dataset_path=dataset_path)
    elif task == "traceability":
        from archbench.tasks.traceability import dataset as trace_dataset
        dataset = trace_dataset.load_dataset(dataset_path=dataset_path, task_type="sad-code")
    else:
        raise ValueError(f"No dataset loader for task: {task}")

    dataset_dict = {d[KEY_INSTANCE_ID]: d for d in dataset}

    # Load predictions
    predictions = load_predictions(predictions_path)
    logger.info(f"Loaded {len(predictions)} predictions")

    # Load evaluation report (aggregated metrics)
    report = None
    if report_path and Path(report_path).exists():
        with open(report_path) as f:
            report = json.load(f)
        logger.info("Loaded evaluation report")
    else:
        # Try to auto-find report next to predictions
        pred_dir = Path(predictions_path).parent.parent
        candidates = list(pred_dir.glob(f"*_{task}_report.json"))
        if candidates:
            with open(candidates[0]) as f:
                report = json.load(f)
            logger.info(f"Auto-found evaluation report: {candidates[0]}")

    # Load per-instance metrics
    instance_metrics = {}
    if instance_metrics_path and Path(instance_metrics_path).exists():
        with open(instance_metrics_path) as f:
            for line in f:
                entry = json.loads(line)
                iid = entry.get(KEY_INSTANCE_ID)
                if iid and entry.get("metrics"):
                    instance_metrics[iid] = entry["metrics"]
        logger.info(f"Loaded per-instance metrics for {len(instance_metrics)} instances")
    else:
        # Auto-find metrics file
        pred_dir = Path(predictions_path).parent.parent
        candidates = list(pred_dir.glob(f"*_{task}_metrics.json"))
        if candidates:
            with open(candidates[0]) as f:
                for line in f:
                    entry = json.loads(line)
                    iid = entry.get(KEY_INSTANCE_ID)
                    if iid and entry.get("metrics"):
                        instance_metrics[iid] = entry["metrics"]
            logger.info(f"Auto-found per-instance metrics: {candidates[0]}")

    # Pick instances that have predictions
    judgeable_ids = [
        iid for iid in dataset_dict
        if iid in predictions and predictions[iid].get(KEY_PREDICTION)
    ]

    # Sample
    random.seed(42)
    actual_sample = min(sample_count, len(judgeable_ids))
    sampled_ids = random.sample(judgeable_ids, actual_sample) if actual_sample < len(judgeable_ids) else judgeable_ids
    sampled_instances = [dataset_dict[iid] for iid in sampled_ids]

    logger.info(f"Sampled {len(sampled_instances)} instances for judge prompt")

    # Build single prompt
    messages = build_prompt(
        sampled_instances=sampled_instances,
        predictions=predictions,
        instance_metrics=instance_metrics,
        report=report,
    )

    # One LLM call
    provider = get_provider(judge_model, ollama_host=ollama_host)
    logger.info(f"Calling {judge_model} for overall assessment...")

    try:
        response_text, api_meta = provider.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        parsed = parse_judge_response(response_text)
    except Exception as e:
        logger.error(f"Judge call failed: {e}")
        parsed = {"score": 0, "summary": f"Error: {e}", "strengths": [], "weaknesses": []}
        api_meta = {}
        response_text = ""

    elapsed = time.time() - start_time

    judge_report = {
        "run_id": run_id,
        "task": task,
        "judge_model": judge_model,
        "timestamp": datetime.now().isoformat(),
        "elapsed_time_seconds": elapsed,
        "predictions_path": predictions_path,
        "instances_sampled": len(sampled_instances),
        "instances_total": len(judgeable_ids),
        "score": parsed["score"],
        "summary": parsed["summary"],
        "strengths": parsed["strengths"],
        "weaknesses": parsed["weaknesses"],
        "examples": parsed["examples"],
        "raw_response": response_text,
        "token_usage": api_meta.get("usage"),
    }

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f"{run_id}_judge_report.json"
    save_report(judge_report, str(report_file))

    # Print
    print_judge_summary(judge_report)

    return judge_report


def print_judge_summary(report: Dict) -> None:
    """Print judge summary to console."""
    print("\n" + "=" * 60)
    print(f"LLM JUDGE ASSESSMENT - {report['task']}")
    print("=" * 60)
    print(f"Judge model: {report['judge_model']}")
    print(f"Instances sampled: {report['instances_sampled']} / {report['instances_total']}")
    print(f"Overall score: {report['score']} / 5")
    print("-" * 60)
    print(f"Summary: {report['summary']}")
    if report["strengths"]:
        print("\nStrengths:")
        for s in report["strengths"]:
            print(f"  + {s}")
    if report["weaknesses"]:
        print("\nWeaknesses:")
        for w in report["weaknesses"]:
            print(f"  - {w}")
    if report.get("examples"):
        print("\nExamples:")
        for ex in report["examples"]:
            print(f"  * {ex}")
    print("-" * 60)
    print(f"Elapsed time: {report['elapsed_time_seconds']:.1f}s")
    print("=" * 60 + "\n")
