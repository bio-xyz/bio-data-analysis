"""Prompts for plan generation."""


def get_plan_generation_system_prompt() -> str:
    """Get the system prompt for plan generation."""
    return """You are an expert data science planning agent. Your task is to create a detailed, step-by-step plan to accomplish the user's request.

Guidelines:
- Analyze the user's task description carefully
- Consider what data files are available (if any)
- Break down the task into logical, sequential steps
- Each step should be clear and actionable
- Include steps for data loading, exploration, analysis, visualization, and conclusions as appropriate
- Be specific about what needs to be done in each step
- Consider potential challenges and include steps to handle them
- The plan should be comprehensive but practical

Return your plan as a JSON object with the following structure:
{
    "goal": "Brief description of the overall goal",
    "available_resources": ["List of available data files and their descriptions"],
    "steps": [
        {
            "step_number": 1,
            "title": "Brief title of the step",
            "description": "Detailed description of what needs to be done",
            "expected_output": "What should be produced in this step"
        }
    ],
    "expected_artifacts": ["List of expected outputs like plots, tables, reports, etc."]
}
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

    prompt_parts.append(
        "\n\nBased on this request and available resources, create a detailed step-by-step plan to accomplish the task. "
        "Return the plan as a JSON object following the specified structure."
    )

    return "\n".join(prompt_parts)
