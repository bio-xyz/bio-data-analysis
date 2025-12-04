"""Prompts for the CODE_PLANNING_NODE - manages step-by-step code execution planning."""

from typing import Optional


def get_code_planning_system_prompt() -> str:
    """Get the system prompt for the code planning node."""
    return """You are an expert data science planning agent responsible for managing step-by-step code execution.
You are being called from a larger system that executes code in discrete steps to achieve a complex task.
Step execution may fail, through no fault of your own, due to code errors, missing libraries, or other issues.
It is okay to encounter failures - your job is to plan the next best action based on the current state.
Main idea is to try to run some small piece of code, see what happens, and then plan the next step accordingly if needed.

Your role is to analyze the current state and decide the next action:

1. ITERATE_CURRENT_STEP - Generate/regenerate code for a step when:
   - No current step exists (first iteration)
   - Previous attempt failed and you want to try a different approach
   - CRITICAL: Need to create a NEW, DISTINCT step goal

2. PROCEED_TO_NEXT_STEP - Move to the next step when:
   - Current step completed successfully
   - Need to plan the next logical step toward the goal
   - CRITICAL: Goal should be specific, achievable in one code cell, and build on previous steps

3. TASK_COMPLETED - Complete the task when:
   - All necessary steps are completed
   - The overall goal has been achieved
   - No more steps are needed
   - CRITICAL: Ensure all user requirements have been met

4. TASK_FAILED - Abort the task when:
   - You've exhausted reasonable approaches and cannot make progress
   - A critical/unrecoverable error occurred that cannot be fixed
   - The task is fundamentally impossible with available resources
   - CRITICAL: Be pragmatic and honest about failures

YOU HAVE FULL AUTONOMY to decide when to TASK_FAILED. Consider:
- If you've tried multiple approaches and none work, TASK_FAILED with explanation
- If there's a critical error (missing data, wrong format, etc.), TASK_FAILED early
- If the task cannot be completed due to fundamental limitations, TASK_FAILED immediately
- Don't keep trying if progress is impossible - be pragmatic

CRITICAL RULES:
- Each step must be SINGLE and ATOMIC (can be done in ONE code cell)
- Each step can be a single action or a small group of related actions: e.g., install library, import library, data loading, cleaning, analysis, visualization
- Steps must build on previous steps logically and progressively toward the overall goal
- Avoid overly complex steps that try to do many things at once
- When iterating on a failed step, generate a COMPLETELY NEW AND DISTINCT approach
- In case of missing libraries, prioritize installing them in the next step and avoid complex workarounds
- NEVER repeat the same failed approach
- Be honest about failures - it's better to TASK_FAILED with partial results than loop forever
- YOU DON'T HAVE AN ACCESS TO THE GPU. FOCUS ON THE TASK AND LIBRARIES THAT CAN BE DONE WITH CPU ONLY
- CPU AND RAM RESOURCES ARE LIMITED, ONLY 2 THREADS AND 4GB OF RAM ARE AVAILABLE
- EACH GOAL MUST COMPLETE WITHIN 2 MINUTES RUN TIME OR ELSE IT IS BEING TIME-OUTED

When generating a new step goal:
- Be specific and actionable
- The step should be achievable in a single code cell
- Consider dependencies on previous steps
- Include clear success criteria

Example simple step goals (these are just examples, exact step goals are very different based on the task):
- Install the pandas, numpy, and matplotlib libraries
- Import the pandas, numpy, and matplotlib libraries
- Load the CSV file 'data.csv' into a pandas DataFrame
- Clean the DataFrame by removing rows with missing values
... etc.

CRITICAL: Return ONLY valid JSON without any markdown formatting or code fences. Markdown is only allowed INSIDE current_step_description field.
"""


def build_code_planning_prompt(
    task_description: str,
    task_rationale: str,
    data_files_description: Optional[str] = None,
    uploaded_files: Optional[list[str]] = None,
    current_step_goal: Optional[str] = None,
    current_step_goal_history: Optional[list[str]] = None,
    last_execution_output: Optional[str] = None,
    last_execution_error: Optional[str] = None,
    completed_steps: Optional[list[dict]] = None,
) -> str:
    """
    Build the user prompt for the code planning node.

    Args:
        task_description: Description of the overall task
        task_rationale: Rationale from the planning node
        data_files_description: Optional description of the data files
        uploaded_files: Optional list of uploaded file names
        current_step_goal: Current step being worked on (if any)
        current_step_goal_history: History of current step goals tried, to avoid repetition
        step_attempts: Number of attempts for current step
        last_execution_output: Output from last execution (if any)
        last_execution_error: Error from last execution (if any)
        completed_steps: List of completed steps with their results

    Returns:
        str: The formatted user prompt
    """
    prompt_parts = [
        "=== OVERALL TASK ===",
        f"Task: {task_description}",
        f"\nTask Rationale: {task_rationale}",
    ]

    if uploaded_files:
        prompt_parts.append(f"\nAvailable Data Files: {', '.join(uploaded_files)}")
        if data_files_description:
            prompt_parts.append(f"Data Files Description: {data_files_description}")

    # Add completed steps
    if completed_steps:
        prompt_parts.append("\n=== COMPLETED STEPS ===")
        for i, step in enumerate(completed_steps, 1):
            prompt_parts.append(f"\nStep {i}: {step.get('goal', 'N/A')}")
            prompt_parts.append(
                f"  Status: {'SUCCESS' if step.get('success') else 'FAILED'}"
            )

    # Add current step information
    if current_step_goal:
        prompt_parts.append("\n=== CURRENT STEP ===")
        prompt_parts.append(f"Goal: {current_step_goal}")
        if current_step_goal_history:
            prompt_parts.append(
                f"Previous Approaches Tried: {', '.join(current_step_goal_history)}"
            )

        if last_execution_error:
            prompt_parts.append(f"\nCURRENT ATTEMPT FAILED!")
            prompt_parts.append(f"Error: {last_execution_error}")
            prompt_parts.append("\nConsider:")
            prompt_parts.append(
                "  - Try a DIFFERENT approach if the error seems fixable"
            )
            prompt_parts.append("  - TASK_FAILED if this is an unrecoverable error")
            prompt_parts.append(
                "  - TASK_FAILED if you've exhausted reasonable alternatives"
            )
        elif last_execution_output:
            prompt_parts.append(
                f"\nCurrent step Execution Output: {last_execution_output[:1000]}"
            )
            prompt_parts.append("\nStep completed successfully!")
    else:
        prompt_parts.append("\n=== CURRENT STEP ===")
        prompt_parts.append("No current step - this is the first iteration.")

    # Add decision guidance
    prompt_parts.append("\n=== DECISION REQUIRED ===")

    if not current_step_goal:
        prompt_parts.append(
            "Since no step has been started, you should ITERATE_CURRENT_STEP with the first step goal."
        )
    elif last_execution_error:
        prompt_parts.append(
            "Previous attempt failed. Analyze the error and decide:\n"
            "  - ITERATE_CURRENT_STEP: Try a NEW, DISTINCT approach (only if you have a viable alternative)\n"
            "  - TASK_FAILED: If the error is unrecoverable or you've exhausted approaches"
        )
    elif last_execution_output or not last_execution_error:
        prompt_parts.append(
            "Previous step succeeded. Decide if you should PROCEED_TO_NEXT_STEP or TASK_COMPLETED if the task is complete."
        )

    prompt_parts.append(
        "\nAnalyze the situation and return your decision as a JSON object."
    )

    return "\n".join(prompt_parts)
