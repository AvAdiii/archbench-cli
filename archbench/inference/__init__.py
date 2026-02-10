"""
ArchBench Inference Module.

This module provides:
- Standardized prompts for each task
- LLM inference with multiple providers (OpenAI, Anthropic, local models)
- Trajectory logging for transparency and verification
"""

from archbench.inference.prompts import (
    create_prompt,
    create_adr_prompt,
    create_adr_prompt_few_shot,
    PROMPT_TEMPLATES,
)

from archbench.inference.run_inference import (
    run_inference,
    run_inference_single,
)

__all__ = [
    "create_prompt",
    "create_adr_prompt",
    "create_adr_prompt_few_shot",
    "PROMPT_TEMPLATES",
    "run_inference",
    "run_inference_single",
]
