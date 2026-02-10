"""
Standardized prompt templates for ArchBench tasks.

These prompts are designed to be:
1. Generic enough for any LLM to understand
2. Consistent across different model providers
3. Clear about expected output format

Users can customize prompts, but these serve as the standard baseline.
"""

from typing import Dict, Optional, List

# =============================================================================
# Few-Shot Examples (from the original ArchAI_ADR paper)
# =============================================================================

FEW_SHOT_EXAMPLES_ADR = [
    {
        "context": """IOG is undertaking a company-wide effort to restructure and standardize its repositories, favoring mono-repos and enforcing shared GitOps and DevOps processes. Parallel to this, a new CI infrastructure is being developed.

Examples of this are:
- input-output-hk/cardano-world
- input-output-hk/ci-world
- input-output-hk/atala-world

This initiative appears to be championed by the SRE team who are the creators of divnix/std. Indeed std is at the heart of the standardization dream.""",

        "decision": """Standardization of the repositories has been deemed a worthwhile endeavour, though of very low priority.

Phase 1 of the standardization process will be carried out in parallel with Move Marconi to a separate repository. A separate repository will be created for Marconi, and from the very beginning it will use std. This way the benefits, limitations and integration costs of std can be experienced and measured, and an informed, definitive decision on standardizing plutus-core and plutus-apps themselves can be made.""",
    },
    {
        "context": """We need to decide on which database management system (DBMS) to use for Project X. The database will be used to store and manage large amounts of data from multiple sources. We need a DBMS that can handle transactions, offer scalability, and provide high reliability and security. Among various options available, we are considering MySQL as a possible choice.

### Decision Considerations
- Ease of use and maintenance
- Community support and resources
- Performance and scalability
- Security and reliability
- Cost and licensing
- Compatibility with our technology stack

### Considered Options
- MySQL
- PostgreSQL
- Oracle
- Microsoft SQL Server
- MongoDB""",

        "decision": """After evaluating the above options based on our decision considerations, we have decided to choose MySQL as our DBMS for Project X.

MySQL is a popular open-source system with a strong development community and a large pool of resources for problem-solving and knowledge sharing. It is well-known for its excellent performance and scalability capabilities, making it ideal for handling vast amounts of data with high levels of efficiency. The platform is secure, reliable, and has a wide range of features that are essential for our project, including ACID compliance for transactions, flexible data model, and support for various programming languages and frameworks.

MySQL is also compatible with the majority of our technology stack, including our web development framework, hosting solutions, and other essential tools. Plus, its cost and licensing terms are competitive compared to other proprietary systems like Oracle and Microsoft SQL Server.""",
    },
]


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_ADR = """You are a software architect assistant specializing in Architecture Decision Records (ADRs).

An ADR documents important architectural decisions made during software development. Given a Context section that describes a situation requiring an architectural decision, your task is to generate an appropriate Decision section.

Your response should:
1. Clearly state the decision made
2. Provide rationale and justification
3. Be specific to the context provided
4. Follow professional ADR conventions

Respond ONLY with the Decision content. Do not include headers like "## Decision" - just provide the decision text directly."""


SYSTEM_PROMPT_ADR_FEW_SHOT = """You are a software architect assistant specializing in Architecture Decision Records (ADRs).

An ADR documents important architectural decisions made during software development. Given a Context section that describes a situation requiring an architectural decision, your task is to generate an appropriate Decision section.

Below are examples of Context-Decision pairs to guide your responses. Follow the same style and level of detail.

Your response should:
1. Clearly state the decision made
2. Provide rationale and justification
3. Be specific to the context provided
4. Follow professional ADR conventions

Respond ONLY with the Decision content. Do not include headers like "## Decision" - just provide the decision text directly."""


# =============================================================================
# ADR Prompts
# =============================================================================

def create_adr_prompt(
    context: str,
    prompt_style: str = "zero_shot",
    include_system_prompt: bool = True,
) -> Dict[str, str]:
    """
    Create a prompt for ADR generation task.

    Args:
        context: The architectural context requiring a decision
        prompt_style: "zero_shot" or "few_shot"
        include_system_prompt: Whether to include a system prompt

    Returns:
        Dictionary with 'system' and 'user' keys for chat-style prompts,
        or just 'prompt' for completion-style models

    Example:
        >>> prompt = create_adr_prompt(
        ...     context="We need to decide on a caching strategy for our API...",
        ...     prompt_style="zero_shot"
        ... )
        >>> print(prompt["user"][:50])
        "## Context\\n\\nWe need to decide on a caching strate"
    """
    if prompt_style == "few_shot":
        return create_adr_prompt_few_shot(context, include_system_prompt)

    # Zero-shot prompt
    user_prompt = f"""## Context

{context.strip()}

## Decision
"""

    if include_system_prompt:
        return {
            "system": SYSTEM_PROMPT_ADR,
            "user": user_prompt,
        }
    else:
        # Completion-style (for models like GPT-3 davinci)
        return {
            "prompt": f"Architectural Decision Record\n\n{user_prompt}",
        }


def create_adr_prompt_few_shot(
    context: str,
    include_system_prompt: bool = True,
) -> Dict[str, str]:
    """
    Create a few-shot prompt for ADR generation with examples.

    Args:
        context: The architectural context requiring a decision
        include_system_prompt: Whether to include a system prompt

    Returns:
        Dictionary with prompt components
    """
    # Build examples section
    examples_text = ""
    for i, example in enumerate(FEW_SHOT_EXAMPLES_ADR, 1):
        examples_text += f"""### Example {i}

## Context

{example["context"].strip()}

## Decision

{example["decision"].strip()}

---

"""

    user_prompt = f"""{examples_text}
### Your Task

## Context

{context.strip()}

## Decision
"""

    if include_system_prompt:
        return {
            "system": SYSTEM_PROMPT_ADR_FEW_SHOT,
            "user": user_prompt,
        }
    else:
        return {
            "prompt": f"Architecture Decision Records - Generate Decision from Context\n\n{user_prompt}",
        }


def create_adr_prompt_chat_format(
    context: str,
    prompt_style: str = "few_shot",
) -> List[Dict[str, str]]:
    """
    Create a chat-formatted prompt with examples as conversation turns.
    Best for chat models like GPT-3.5-turbo and GPT-4.

    Args:
        context: The architectural context
        prompt_style: "zero_shot" or "few_shot"

    Returns:
        List of message dictionaries for chat completion API
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_ADR if prompt_style == "zero_shot" else SYSTEM_PROMPT_ADR_FEW_SHOT,
        }
    ]

    if prompt_style == "few_shot":
        # Add examples as conversation turns
        for example in FEW_SHOT_EXAMPLES_ADR:
            messages.append({
                "role": "user",
                "content": f"## Context\n\n{example['context'].strip()}",
            })
            messages.append({
                "role": "assistant",
                "content": example["decision"].strip(),
            })

    # Add the actual query
    messages.append({
        "role": "user",
        "content": f"## Context\n\n{context.strip()}",
    })

    return messages


# =============================================================================
# Traceability Prompts
# =============================================================================

SYSTEM_PROMPT_TRACEABILITY = """You are a software architecture expert specializing in traceability link recovery.

Your task is to identify trace links between software architecture documentation and source code artifacts. A trace link connects a sentence or section in the documentation to the code files that implement or relate to that documentation.

For each documentation sentence provided, identify which code files are related to it. Return your answer as a JSON array of objects, where each object has:
- "doc_sentence": The sentence number from the documentation
- "code_artifact": The path to the related code file
- "confidence": Your confidence score (0.0 to 1.0)

Only include links where you are reasonably confident of the relationship."""


def create_traceability_prompt(
    documentation: str,
    code_artifacts: List[str],
    include_system_prompt: bool = True,
) -> Dict[str, str]:
    """
    Create a prompt for traceability link recovery task.

    Args:
        documentation: The architecture documentation text
        code_artifacts: List of code file paths
        include_system_prompt: Whether to include a system prompt

    Returns:
        Dictionary with prompt components
    """
    # Number the sentences for reference
    sentences = documentation.split(". ")
    numbered_doc = "\n".join([f"{i+1}. {s.strip()}" for i, s in enumerate(sentences) if s.strip()])

    code_list = "\n".join([f"- {artifact}" for artifact in code_artifacts])

    user_prompt = f"""## Architecture Documentation

{numbered_doc}

## Code Artifacts

{code_list}

## Task

Identify which documentation sentences relate to which code artifacts. Return your answer as a JSON array.

Example output format:
```json
[
  {{"doc_sentence": 1, "code_artifact": "src/storage/MediaStore.java", "confidence": 0.95}},
  {{"doc_sentence": 3, "code_artifact": "src/api/MediaController.java", "confidence": 0.82}}
]
```

Your response (JSON only):"""

    if include_system_prompt:
        return {
            "system": SYSTEM_PROMPT_TRACEABILITY,
            "user": user_prompt,
        }
    else:
        return {
            "prompt": user_prompt,
        }


# =============================================================================
# Generic Prompt Interface
# =============================================================================

PROMPT_TEMPLATES = {
    "adr": {
        "zero_shot": create_adr_prompt,
        "few_shot": create_adr_prompt_few_shot,
    },
    "traceability": {
        "zero_shot": create_traceability_prompt,
    },
    # TODO: Add serverless and dynamic prompts
}


def create_prompt(
    task: str,
    instance: Dict,
    prompt_style: str = "zero_shot",
    **kwargs,
) -> Dict[str, str]:
    """
    Create a prompt for any ArchBench task.

    Args:
        task: Task name (adr, traceability, serverless, dynamic)
        instance: Dataset instance dictionary
        prompt_style: Prompt style (zero_shot, few_shot, cot)
        **kwargs: Additional arguments for specific tasks

    Returns:
        Dictionary with prompt components
    """
    if task == "adr":
        return create_adr_prompt(
            context=instance["context"],
            prompt_style=prompt_style,
            **kwargs,
        )
    elif task == "traceability":
        return create_traceability_prompt(
            documentation=instance["documentation"],
            code_artifacts=instance["code_artifacts"],
            **kwargs,
        )
    else:
        raise ValueError(f"Prompt not implemented for task: {task}")
