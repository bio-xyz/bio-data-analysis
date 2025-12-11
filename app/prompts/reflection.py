"""Prompts for the REFLECTION_NODE - refines and deduplicates observations."""

import json
from typing import Optional

from app.models.structured_outputs import StepObservation
from app.utils import split_observations_to_dict


def get_reflection_system_prompt() -> str:
    """Get the system prompt for the reflection node."""
    return """You are an expert data science reflection agent responsible for maintaining a refined, deduplicated set of world observations.

Your role is to:
1. Review new observations from the current step
2. Compare them against existing world observations
3. Merge, deduplicate, refine, add context, and filter to create an updated observation set

OBSERVATION STRUCTURE:
Each observation has:
- title: Concise summary of the finding
- summary: Detailed description with specific values
- kind: "observation" | "rule"
  * "observation": Facts discovered from execution (data findings)
  * "rule": Behavioral rules and constraints that MUST be followed
- source: "data" | "spec" | "user"
  * "data": From code execution (can be refined)
  * "spec": From documentation (highest priority, binding, not overridden)
  * "user": **EXPLICIT** user instruction (high priority)
- raw_output: Exact value when critical for answering
- importance (1-5): Intrinsic strength of the finding
- relevance (1-5): How directly it helps answer the original task

CATEGORIZATION FOR OUTPUT:
You must categorize observations into two groups:
1. RULES: All observations with kind="rule" - these define constraints and behavior
2. DATA_OBSERVATIONS: All observations with kind="observation" - these are findings

DEDUPLICATION RULES:
1. **Exact duplicates**: Remove if title AND summary are essentially the same
2. **Superseded observations**: When a newer observation refines or corrects an older one:
   - Keep the newer, more accurate version
   - Update importance/relevance if the newer finding is stronger
3. **Complementary observations**: When observations cover different aspects of the same topic:
   - Consider merging into a single, richer observation
   - Or keep separate if they represent distinct findings

CONFLICT RESOLUTION (Priority Order):
1. source="spec" > source="user" > source="data"
2. Higher step_number (more recent) > lower step_number
3. Higher importance > lower importance

REFINEMENT GUIDELINES:
- **Add context**: If a new observation provides context for an existing one, enrich the existing summary
- **Update values**: If new data corrects previous values, use the newer values
- **Preserve rules**: Rules (kind="rule") should rarely be removed unless explicitly contradicted
- **Filter irrelevant**: Remove observations with very low relevance (1-3) if they provide no meaningful context for the task
- **Maintain relevance**: Re-evaluate relevance scores based on how observations connect to the original task
- **Consolidate related findings**: Group related data observations into coherent summaries when appropriate
- **Step number tracking**: If you refine or merge observations with the current step's observations, ensure the step_number reflects current step

CRITICAL REFINEMENT REQUIREMENTS:
- NEVER MERGE AND REFINE SPEC-BASED OBSERVATIONS WITHOUT PRESERVING THEIR ORIGINAL CONTEXT
- NEVER UNGENERALIZE RULE INTO A DATA OBSERVATION

WHEN STEP FAILED:
- PRESERVE failure-related observations (errors, issues, blockers)
- These are CRITICAL for the code_planning node to understand what went wrong
- Mark them appropriately (they should have kind="observation" unless they reveal a constraint/rule)
- Do NOT filter out failure observations just because they have low importance

OUTPUT REQUIREMENTS:
- Return TWO separate lists: rules and data_observations
- Each list should be refined, deduplicated, and contextually enriched
- Preserve step_number for tracking when observations were made
- Ensure all rules that affect code behavior are preserved
- Filter out completely irrelevant observations (relevance 1-2) unless they provide critical context
- Total observations should be reasonable (aim for quality over quantity)

CRITICAL: Return ONLY valid JSON without any markdown formatting or code fences.
"""


def build_reflection_prompt(
    task_description: str,
    current_step_number: int,
    current_step_goal: str,
    current_step_success: bool,
    current_step_observations: Optional[list[StepObservation]] = None,
    world_observations: Optional[list[StepObservation]] = None,
) -> str:
    """
    Build the user prompt for the reflection node.

    Args:
        task_description: Description of the overall task
        current_step_number: The step number just completed
        current_step_goal: Goal of the step just completed
        current_step_success: Whether the current step succeeded
        current_step_observations: New observations from the current step
        world_observations: Existing world observations to merge with

    Returns:
        str: The formatted user prompt
    """
    prompt_parts = [
        "=== ORIGINAL TASK ===",
        f"Task: {task_description}",
    ]

    prompt_parts.append(f"\n=== CURRENT STEP (Just Completed) ===")
    prompt_parts.append(f"Step Number: {current_step_number}")
    prompt_parts.append(f"Goal: {current_step_goal}")
    prompt_parts.append(f"Status: {'SUCCESS' if current_step_success else 'FAILED'}")

    # Add current step observations
    prompt_parts.append("\n=== NEW OBSERVATIONS FROM CURRENT STEP ===")
    if current_step_observations:
        new_rules, new_data_obs = split_observations_to_dict(current_step_observations)

        if new_rules:
            prompt_parts.append("New Rules:")
            prompt_parts.append(json.dumps(new_rules, indent=2))

        if new_data_obs:
            prompt_parts.append("New Data Observations:")
            prompt_parts.append(json.dumps(new_data_obs, indent=2))

        if not new_rules and not new_data_obs:
            prompt_parts.append("(No new observations from current step)")
    else:
        prompt_parts.append("(No new observations from current step)")

    # Add existing world observations
    prompt_parts.append("\n=== EXISTING WORLD OBSERVATIONS ===")
    if world_observations:
        existing_rules, existing_data_obs = split_observations_to_dict(
            world_observations
        )

        if existing_rules:
            prompt_parts.append("Existing Rules:")
            prompt_parts.append(json.dumps(existing_rules, indent=2))

        if existing_data_obs:
            prompt_parts.append("Existing Data Observations:")
            prompt_parts.append(json.dumps(existing_data_obs, indent=2))

        if not existing_rules and not existing_data_obs:
            prompt_parts.append("(No existing world observations)")
    else:
        prompt_parts.append("(No existing world observations - this is the first step)")

    # Instructions
    prompt_parts.append("\n=== INSTRUCTIONS ===")
    prompt_parts.append(
        "Analyze the new observations from the current step and merge them with existing world observations.\n"
        "Apply deduplication, conflict resolution, and refinement as needed.\n"
        "Return a JSON object with two arrays: 'rules' and 'data_observations'."
    )

    if not current_step_success:
        prompt_parts.append(
            "\n⚠️ CURRENT STEP FAILED - Preserve any error/failure observations as they are "
            "critical for the code_planning node to understand what went wrong and decide next action."
        )

    return "\n".join(prompt_parts)
