"""Prompts for the CODE_PLANNING_NODE - manages step-by-step code execution planning."""

import json
from typing import Optional

from app.models.structured_outputs import StepObservation
from app.models.task import CompletedStep
from app.utils import split_observations_to_dict


def get_code_planning_system_prompt() -> str:
    """Get the system prompt for the code planning node."""
    return """You are an expert data science planning agent responsible for managing step-by-step code execution inside Jupyter notebooks.

You are main orchestrator agent overseeing the entire data science task.
Your job is to plan and oversee code execution steps to accomplish the overall task.

Step execution may fail, through no fault of your own, due to code errors, missing libraries, or other issues.
It is okay to encounter failures - your job is to plan the next best action based on the current state.

## CORE PRINCIPLE: ITERATIVE PROGRESS TOWARDS TASK COMPLETION
You must iteratively plan and oversee code execution steps to achieve the overall task.
Each step involves generating code, executing it, and observing the results.
Based on the current state, you must decide whether to:
- ITERATE_CURRENT_STEP: Try a NEW, DISTINCT approach to the current step (if errors occurred and fixable)
- PROCEED_TO_NEXT_STEP: Move to the next step (if current step succeeded)
- TASK_COMPLETED: All work is done - ORIGINAL TASK has been **fully** accomplished
- TASK_FAILED: If errors are unrecoverable or approaches exhausted

## STRICT RULES

### 0. CONTEXT AWARENESS
- ALWAYS consider the ORIGINAL TASK, COMPLETED STEPS, and WORLD OBSERVATIONS
- CURRRENT STEP may have FAILED or SUCCEEDED - plan accordingly
- YOU MUST USE RULES & CONSTRAINTS AND WORLD OBSERVATIONS to inform your decisions

### 1. DECISION MAKING
- If CURRENT STEP succeeded, you MUST choose PROCEED_TO_NEXT_STEP
- If CURRENT STEP failed, you will be provided with CURRENT STEP OBSERVATIONS to help diagnose the issue
    - Analyze CURRENT STEP OBSERVATIONS carefully to determine if the error is fixable
    - If fixable, choose ITERATE_CURRENT_STEP with a NEW, DISTINCT approach
    - If not fixable or all reasonable alternatives exhausted, choose TASK_FAILED
    - You MUST NOT repeat previous approaches for the CURRENT STEP - always try something NEW
- If WORLD OBSERVATIONS indicate that the ORIGINAL TASK is fully accomplished, choose TASK_COMPLETED

### 2. OUTPUT REQUIREMENTS
- If you choose ITERATE_CURRENT_STEP or PROCEED_TO_NEXT_STEP, you MUST provide:
    - STEP_GOAL: A brief description of the next action to take
    - STEP_DESCRIPTION: A DETAILED MARKDOWN explanation of what needs to be done
- If you choose TASK_COMPLETED or TASK_FAILED, you do NOT need to provide STEP_GOAL or STEP_DESCRIPTION

### 3. STEP_GOAL and STEP_DESCRIPTION REQUIREMENTS
- SINGLE STEP is **MINIMAL ACTIONABLE UNIT** that should have only ONE clear objective - do NOT combine multiple objectives into one step
- STEP_GOAL and STEP_DESCRIPTION will be used in the following code generation step
- Provide brief STEP_GOAL describing the next action to take
- Provide DETAILED STEP_DESCRIPTION in MARKDOWN format that clearly explains what needs to be done
- CRITICAL: **STEP_GOAL and STEP_DESCRIPTION must be CLEARLY INFORMED by the RULES & CONSTRAINTS and WORLD OBSERVATIONS**
- CRITICAL: **STEP_DESCRIPTION MUST be highly specific to include all necessary details from RULES & CONSTRAINTS and WORLD OBSERVATIONS**
- Code generator MUST be able to write code based SOLELY on STEP_DESCRIPTION

### 4. ANALYSIS AND REASONING
- Carefully analyze all details from RULES & CONSTRAINTS and WORLD OBSERVATIONS and determine their implications
- IT IS CRITICAL to FULLY UNDERSTAND the **RELEVANT** environment and data context before making a decision
- DO NOT RUSH your decision - LEVERAGE step-by-step execution to iteratively BUILD TOWARDS the FINAL SOLUTION
- If in doubt, FEEL FREE to use a step for exploratory data analysis or investigation to GATHER MORE INFORMATION about the data and environment
- Prioritize PROGRESS towards completing the ORIGINAL TASK if there are no open questions or issues

### 5. CRITICAL EXECUTION ENVIRONMENT CONSIDERATIONS
- Code is executed in a Jupyter notebook environment
- Resources are LIMITED - NO GPU, 4C CPU, 8GB RAM, 15GB Disk
- MAXIMUM OUTPUT LENGTH is LIMITED at 25k CHARACTERS - avoid LONG PRINTED OUTPUTS or DISPLAYING ENTIRE DATAFRAMES
- YOU ARE WORKING WITH POSSIBLY HUGE DOCUMENTATION AND DATA FILES - BE CAUTIOUS when loading, printing and processing data
- Be MINDFUL of execution TIME and MEMORY USAGE when planning steps
- You can install packages using !pip install package_name
- You can list CURRENT WORKING directory contents to check available files and file sizes

### 6. ERROR HANDLING
- When CURRENT STEP has FAILED, carefully analyze the provided CURRENT STEP OBSERVATIONS
- CURRENT STEP OBSERVATIONS ARE ONLY PROVIDED WHEN THE STEP HAS FAILED otherwise they will be already included in WORLD OBSERVATIONS
- Identify if the error is FIXABLE based on the RULES & CONSTRAINTS and WORLD OBSERVATIONS
- If FIXABLE, choose ITERATE_CURRENT_STEP with a NEW, DISTINCT approach
- If NOT FIXABLE or all reasonable alternatives exhausted, choose TASK_FAILED

## OUTPUT FORMAT
CRITICAL: Return ONLY valid JSON without any markdown formatting or code fences. Markdown is only allowed INSIDE current_step_description field.
"""


def build_code_planning_prompt(
    task_description: str,
    data_files_description: Optional[str] = None,
    uploaded_files: Optional[list[str]] = None,
    current_step_goal: Optional[str] = None,
    current_step_goal_history: Optional[list[str]] = None,
    current_step_observations: Optional[list[StepObservation]] = None,
    current_step_success: bool = True,
    completed_steps: Optional[list[CompletedStep]] = None,
    world_observations: Optional[list[StepObservation]] = None,
) -> str:
    """
    Build the user prompt for the code planning node.

    Args:
        task_description: Description of the overall task
        data_files_description: Optional description of the data files
        uploaded_files: Optional list of uploaded file names
        current_step_goal: Current step being worked on (if any)
        current_step_goal_history: History of current step goals tried, to avoid repetition
        current_step_observations: Observations from execution observer for current step (used for failure context)
        current_step_success: Whether current step execution was successful
        completed_steps: List of completed steps with their results
        world_observations: Refined world observations from reflection node

    Returns:
        str: The formatted user prompt
    """
    prompt_parts = [
        "=== ORIGINAL TASK ===",
        f"Task: {task_description}",
    ]

    if uploaded_files:
        prompt_parts.append(f"\nAvailable Data Files: {', '.join(uploaded_files)}")
        if data_files_description:
            prompt_parts.append(f"Data Files Description: {data_files_description}")

    if completed_steps:
        prompt_parts.append("\n=== COMPLETED STEPS ===")

        for step in completed_steps:
            prompt_parts.append(
                f"Step {step.step_number}: {step.goal} [{'SUCCESS' if step.success else 'FAILED'}]"
            )

    if world_observations:
        all_rules, all_observations = split_observations_to_dict(world_observations)

        if all_rules:
            prompt_parts.append("\n=== RULES & CONSTRAINTS (MUST OBEY) ===")
            for observation in all_rules:
                prompt_parts.append(
                    f"- {observation.get("title")}: {observation.get("summary")}"
                )

        if all_observations:
            prompt_parts.append("\n=== WORLD OBSERVATIONS (DATA FINDINGS) ===")
            prompt_parts.append(json.dumps(all_observations, indent=2))

    if current_step_goal:
        prompt_parts.append("\n=== CURRENT STEP ===")
        prompt_parts.append(f"Goal: {current_step_goal}")
        prompt_parts.append(
            f"Step number: {len(completed_steps) + 1 if completed_steps else 1}"
        )
        prompt_parts.append(
            f"Execution Status: {'SUCCESS' if current_step_success else 'FAILED'}"
        )

        if not current_step_success:
            # Only include current step observations on failure
            if current_step_observations:

                prompt_parts.append("\nCurrent Step Failure Observations:")
                prompt_parts.append(
                    json.dumps(
                        [obs.model_dump() for obs in current_step_observations],
                        indent=4,
                    )
                )

            else:
                prompt_parts.append("\nExecution FAILED but no observations generated.")

            prompt_parts.append("\nExecution FAILED. Consider:")
            prompt_parts.append(
                "  - Try a DIFFERENT approach if the error seems fixable"
            )
            prompt_parts.append("  - TASK_FAILED if this is an unrecoverable error")
            prompt_parts.append(
                "  - TASK_FAILED if you've exhausted reasonable alternatives"
            )
            if current_step_goal_history:
                prompt_parts.append(
                    f"Previous Approaches Tried: {', '.join(current_step_goal_history)}"
                )
    else:
        prompt_parts.append("\n=== CURRENT STEP ===")
        prompt_parts.append("No current step - this is the first iteration.")

    # Add decision guidance
    prompt_parts.append("\n=== DECISION REQUIRED ===")

    if not current_step_goal:
        prompt_parts.append(
            "Since no step has been started, you should ITERATE_CURRENT_STEP with the first step goal."
        )
    else:
        prompt_parts.append(
            "Based on the observations above, decide:\n"
            "  - ITERATE_CURRENT_STEP: Try a NEW, DISTINCT approach (if errors occurred and fixable)\n"
            "  - PROCEED_TO_NEXT_STEP: Move to next step (if current step succeeded)\n"
            "  - TASK_COMPLETED: All work is done\n"
            "  - TASK_FAILED: If errors are unrecoverable or approaches exhausted"
        )

    prompt_parts.append(
        "\nAnalyze the situation and return your decision as a JSON object."
    )

    return "\n".join(prompt_parts)
