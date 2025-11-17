"""Prompts for code generation."""


def get_code_generation_system_prompt() -> str:
    """Get the system prompt for code generation."""
    return """You are an expert data science code generator. Your task is to generate Python code that accomplishes the user's task.

Guidelines:
- Generate clean, well-commented Python code
- Use pandas, numpy, matplotlib, seaborn, and other common data science libraries as needed
- Include proper error handling
- Generate code that can be executed in a Jupyter notebook environment
- If data files are provided, assume they are available in the current directory
- Output should be complete and executable code
- Do not include markdown formatting or code fences - return only the Python code
- If previous code failed with an error, analyze the error and fix the issue in the new code
- Common fixes: add missing imports, correct variable names, fix syntax errors, handle edge cases
"""


def build_code_generation_prompt(
    task_description: str,
    data_files_description: str | None = None,
    uploaded_files: list[str] | None = None,
    plan: str | None = None,
    previous_code: str | None = None,
    previous_error: str | None = None,
) -> str:
    """
    Build the user prompt for code generation.

    Args:
        task_description: Description of the task to accomplish
        data_files_description: Optional description of the data files
        uploaded_files: Optional list of uploaded file names
        plan: Optional step-by-step plan to follow
        previous_code: Optional previous code that failed
        previous_error: Optional error message from previous execution

    Returns:
        str: The formatted user prompt
    """
    prompt_parts = [f"Task: {task_description}"]

    if uploaded_files:
        prompt_parts.append(f"\nAvailable data files: {', '.join(uploaded_files)}")

    if data_files_description:
        prompt_parts.append(f"\nData files description: {data_files_description}")

    if plan:
        prompt_parts.append(f"\n\nPlan to follow:\n{plan}")

    # Add error context if this is a retry
    if previous_code and previous_error:
        prompt_parts.append("\n\CRITICAL: PREVIOUS ATTEMPT FAILED")
        prompt_parts.append(f"\nPrevious code:\n```python\n{previous_code}\n```")
        prompt_parts.append(f"\nError encountered:\n{previous_error}")
        prompt_parts.append("\nPlease fix the error and generate corrected code.")

    prompt_parts.append(
        "\n\nGenerate Python code to accomplish this task following the plan. Return only the executable Python code without any markdown formatting or explanations."
    )

    return "\n".join(prompt_parts)
