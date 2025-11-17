"""Prompts for plan generation."""


def get_plan_generation_system_prompt() -> str:
    """Get the system prompt for plan generation."""
    return """You are an expert data science planning agent. Your task is to create a detailed, step-by-step plan to accomplish the user's request.

Guidelines:
- Analyze the user's task description carefully
- ONLY use the data files that are explicitly listed as available - do NOT assume or invent file names
- Determine if the task can be completed with the information provided:
  * If the task description contains all necessary data (e.g., "analyze word strawberry", "calculate [1,2,3,4]"), proceed with planning
  * If the task references external data that wasn't provided (e.g., "given this CSV", "analyze the uploaded file", "use the dataset"), mark as incomplete
- Break down the task into logical, sequential steps
- Each step should be clear and actionable
- Include steps for data loading, exploration, analysis, visualization, and conclusions as appropriate
- Be specific about what needs to be done in each step
- Consider potential challenges and include steps to handle them
- The plan should be comprehensive but practical

Return your plan as a JSON object following this structure:

{
    "success": true or false,
    "goal": "Brief description of the overall goal (empty string if success=false)",
    "available_resources": ["List of ONLY the actual data files that were provided - leave empty if none"],
    "steps": [
        {
            "step_number": 1,
            "title": "Brief title of the step",
            "description": "Detailed description of what needs to be done",
            "expected_output": "What should be produced in this step"
        }
    ],
    "expected_artifacts": ["List of expected outputs like plots, tables, reports, etc. - empty if success=false"],
    "error": "Clear explanation of what data/files are missing (empty string if success=true)"
}

If the task CAN be completed, set success=true and fill in goal, steps, expected_artifacts.
If the task CANNOT be completed due to missing data, set success=false and provide a clear error message explaining what's needed.
"""


def build_plan_generation_prompt(
    task_description: str,
    data_files_description: str | None = None,
    uploaded_files: list[str] | None = None,
) -> str:
    """
    Build the user prompt for plan generation.

    Args:
        task_description: Description of the task to accomplish
        data_files_description: Optional description of the data files
        uploaded_files: Optional list of uploaded file names

    Returns:
        str: The formatted user prompt
    """
    prompt_parts = [f"User Request: {task_description}"]

    if uploaded_files:
        prompt_parts.append(f"\nAvailable Data Files: {', '.join(uploaded_files)}")
        if data_files_description:
            prompt_parts.append(f"\nData Files Description: {data_files_description}")
    else:
        prompt_parts.append("\nAvailable Data Files: NONE - No files have been uploaded.")

    prompt_parts.append(
        "\n\nIMPORTANT: "
        "Carefully evaluate if the task can be completed with the available information. "
        "If the task references external data (e.g., 'this CSV', 'the dataset', 'uploaded file') that is not in the available files list, "
        "return success=false with an error message requesting the missing data. "
        "If all necessary data is either provided in the available files OR contained within the task description itself (e.g., specific values, words, numbers), "
        "return success=true with a complete plan. "
        "Do NOT create synthetic/sample data to substitute for referenced but missing files. "
        "Return the plan as a JSON object following the specified structure."
    )

    return "\n".join(prompt_parts)
