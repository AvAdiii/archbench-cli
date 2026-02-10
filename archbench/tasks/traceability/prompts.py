"""
Traceability Link Recovery (TLR) prompt templates.

Provides prompts for SAD-SAM and SAD-Code traceability tasks.
"""

from typing import Dict, List, Optional


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_SAD_CODE = """You are a software architecture expert specializing in traceability link recovery.

Your task is to identify trace links between Software Architecture Documentation (SAD) and source code. A trace link connects a sentence in the documentation to code files that implement or relate to that sentence.

For each sentence in the documentation, identify which code files are related. Return your answer as a JSON array of objects with:
- "sentence": The sentence number from the documentation
- "target": The code file path

Only include links where you are confident of the relationship.

Example output:
[
  {"sentence": 1, "target": "src/main/Facade.java"},
  {"sentence": 7, "target": "src/business/MediaManagement.java"}
]"""


SYSTEM_PROMPT_SAD_SAM = """You are a software architecture expert specializing in traceability link recovery.

Your task is to identify trace links between Software Architecture Documentation (SAD) and Software Architecture Model (SAM) elements. A trace link connects a sentence in the documentation to model elements (components, interfaces, etc.).

For each sentence in the documentation, identify which model elements are related. Return your answer as a JSON array of objects with:
- "sentence": The sentence number from the documentation
- "target": The model element ID

Only include links where you are confident of the relationship.

Example output:
[
  {"sentence": 1, "target": "Component_Facade"},
  {"sentence": 7, "target": "Component_MediaManagement"}
]"""


# =============================================================================
# Prompt Creation Functions
# =============================================================================

def format_sentences(sentences: List[str]) -> str:
    """
    Format sentences with line numbers for the prompt.

    Args:
        sentences: List of sentence strings (index 0 should be empty)

    Returns:
        Formatted string with numbered sentences
    """
    # Skip index 0 (empty), start from 1
    formatted = []
    for i in range(1, len(sentences)):
        formatted.append(f"{i}. {sentences[i]}")

    return "\n".join(formatted)


def create_prompt_sad_code(
    sentences: List[str],
    code_files: Optional[List[str]] = None,
    prompt_style: str = "zero_shot",
) -> Dict[str, str]:
    """
    Create prompt for SAD-Code traceability.

    Args:
        sentences: List of sentences from SAD (index 0 is empty, sentences start at 1)
        code_files: List of available code files (REQUIRED for exact matching)
        prompt_style: "zero_shot" (default)

    Returns:
        Dictionary with 'system' and 'user' keys
    """
    formatted_sentences = format_sentences(sentences)

    if not code_files:
        raise ValueError("code_files must be provided for SAD-Code traceability")

    # List all available code files
    code_list = "\n".join([f"- {f}" for f in sorted(code_files)])

    user_prompt = f"""## Software Architecture Documentation

{formatted_sentences}

## Available Code Files

{code_list}

## Task

For each sentence in the documentation, identify which code files (from the list above) implement or relate to that sentence.

**IMPORTANT**: Only use file paths exactly as they appear in the "Available Code Files" list above.

Return a JSON array of trace links:
[
  {{"sentence": <number>, "target": "<exact_file_path_from_list>"}}
]

Your response (JSON only):"""

    return {
        "system": SYSTEM_PROMPT_SAD_CODE,
        "user": user_prompt,
    }


def create_prompt_sad_sam(
    sentences: List[str],
    model_elements: Optional[List[str]] = None,
    prompt_style: str = "zero_shot",
) -> Dict[str, str]:
    """
    Create prompt for SAD-SAM traceability.

    Args:
        sentences: List of sentences from SAD (index 0 is empty, sentences start at 1)
        model_elements: Optional list of model element IDs
        prompt_style: "zero_shot" (default)

    Returns:
        Dictionary with 'system' and 'user' keys
    """
    formatted_sentences = format_sentences(sentences)

    if model_elements:
        # If model elements are provided, list them
        elements_list = "\n".join([f"- {e}" for e in sorted(model_elements)])
        user_prompt = f"""## Software Architecture Documentation

{formatted_sentences}

## Model Elements

{elements_list}

## Task

Identify which sentences in the documentation relate to which model elements. Return a JSON array of trace links.

Your response (JSON only):"""
    else:
        # No model elements provided - LLM infers from sentence content
        user_prompt = f"""## Software Architecture Documentation

{formatted_sentences}

## Task

Based on the documentation, identify which sentences describe specific architectural components or model elements. Infer likely model element names from the components described.

Return a JSON array of trace links with this format:
[
  {{"sentence": <number>, "target": "<inferred_element_name>"}}
]

Your response (JSON only):"""

    return {
        "system": SYSTEM_PROMPT_SAD_SAM,
        "user": user_prompt,
    }


def create_chat_messages(
    sentences: List[str],
    task_type: str = "sad-code",
    code_files: Optional[List[str]] = None,
    model_elements: Optional[List[str]] = None,
    prompt_style: str = "zero_shot",
) -> List[Dict[str, str]]:
    """
    Create chat-formatted prompt for traceability.

    Args:
        sentences: List of sentences from SAD
        task_type: "sad-code" or "sad-sam"
        code_files: List of code files (REQUIRED for sad-code)
        model_elements: List of model elements (REQUIRED for sad-sam)
        prompt_style: "zero_shot" (default)

    Returns:
        List of message dictionaries
    """
    if task_type == "sad-code":
        if not code_files:
            raise ValueError("code_files must be provided for sad-code traceability")
        prompt_dict = create_prompt_sad_code(sentences, code_files, prompt_style)
    elif task_type == "sad-sam":
        if not model_elements:
            raise ValueError("model_elements must be provided for sad-sam traceability")
        prompt_dict = create_prompt_sad_sam(sentences, model_elements, prompt_style)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return [
        {"role": "system", "content": prompt_dict["system"]},
        {"role": "user", "content": prompt_dict["user"]},
    ]
