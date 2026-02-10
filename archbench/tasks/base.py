#!/usr/bin/env python3
"""
Base interfaces for ArchBench tasks.

Each task must implement:
- Data loading from datasets
- Prompt generation for LLM inference
- Evaluation metrics computation
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional


class TaskBase(ABC):
    """Base interface for all ArchBench tasks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Task identifier (e.g., 'adr', 'traceability')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable task description."""
        pass

    @abstractmethod
    def load_dataset(
        self,
        dataset_path: Optional[str] = None,
        instance_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load dataset instances.

        Args:
            dataset_path: Path to dataset file/directory
            instance_ids: Optional filter for specific instances

        Returns:
            List of dataset instances with standardized keys:
            - instance_id: Unique identifier
            - Additional task-specific fields
        """
        pass

    @abstractmethod
    def create_prompt(
        self,
        instance: Dict[str, Any],
        prompt_style: str = "zero_shot",
    ) -> List[Dict[str, str]]:
        """
        Create LLM prompt for an instance.

        Args:
            instance: Dataset instance
            prompt_style: Prompting strategy (zero_shot, few_shot, etc.)

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        pass

    @abstractmethod
    def extract_prediction(self, raw_output: str) -> Any:
        """
        Extract structured prediction from raw LLM output.

        Args:
            raw_output: Raw text from LLM

        Returns:
            Extracted prediction (format depends on task)
        """
        pass

    @abstractmethod
    def compute_metrics(
        self,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: List of predictions with instance_id and prediction
            references: List of ground truth references
            **kwargs: Task-specific options (e.g., skip_bertscore)

        Returns:
            Dictionary of metric names to scores
        """
        pass
