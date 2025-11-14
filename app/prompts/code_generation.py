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
"""


def build_code_generation_prompt(
    task_description: str,
    data_files_description: str | None = None,
    uploaded_files: list[str] | None = None,
) -> str:
    """
    Build the user prompt for code generation.

    Args:
        task_description: Description of the task to accomplish
        data_files_description: Optional description of the data files
        uploaded_files: Optional list of uploaded file names

    Returns:
        str: The formatted user prompt
    """
    prompt_parts = [f"Task: {task_description}"]

    if uploaded_files:
        prompt_parts.append(f"\nAvailable data files: {', '.join(uploaded_files)}")

    if data_files_description:
        prompt_parts.append(f"\nData files description: {data_files_description}")

    prompt_parts.append(
        "\n\nGenerate Python code to accomplish this task. Return only the executable Python code without any markdown formatting or explanations."
    )

    return "\n".join(prompt_parts)
