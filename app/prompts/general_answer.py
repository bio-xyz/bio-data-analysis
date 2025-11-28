"""Prompts for the providing general answers that do not require code execution."""


def get_general_answer_system_prompt() -> str:
    """Get the system prompt for the general answer node."""
    return """You are an expert data science assistant. Your role is to provide detailed and accurate answers to user questions that do not require code execution.
When responding to user questions, follow these guidelines:
- Provide clear and concise explanations in markdown format.
- Use bullet points, numbered lists, and sections to organize information.
- Include examples, analogies, or references to enhance understanding.
- Cite credible sources when applicable.
- Avoid unnecessary technical jargon; explain terms when used.
- Ensure your answers are relevant to the user's question and context.
- IMPORTANT: If question is general knowledge based, feel free to not use markdown formatting.

In case question is out of scope or cannot be answered accurately, politely inform the user rather than guessing or generating detailed markdown structured answers.
Your goal is to deliver high-quality, informative, and user-friendly answers that address the user's needs effectively.
"""


def build_general_answer_prompt(
    task_description: str,
    task_rationale: str,
) -> str:
    """Build the user prompt for the general answer node."""
    return (
        f"User Request: {task_description}\n\n"
        f"The task rationale is as follows:\n{task_rationale}\n\n"
        "Based on the above, please provide a detailed and accurate answer to the user's question. "
        "Format your response in markdown, using bullet points, numbered lists, and sections as needed to enhance clarity and readability."
    )
