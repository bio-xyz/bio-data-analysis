"""Prompts for the generating clarification questions when the task is ambiguous."""


def get_task_clarification_system_prompt() -> str:
    """Get the system prompt for the clarification node."""
    return """You are an expert data science agent tasked with clarifying ambiguous user requests. Your role is to identify missing information and ask specific questions to help the user refine their request.
    
When generating clarification questions, consider the following:
- What specific details are needed to understand the user's intent?
- Are there any assumptions that need to be validated?
- What constraints or preferences should be clarified?
- Are there any technical requirements or limitations that need to be addressed?

Generate 1-5 clear and concise clarification questions that will help the user provide the necessary information to proceed with their request effectively.
"""


def build_task_clarification_prompt(
    task_description: str,
    task_rationale: str,
):
    """
    Build the user prompt for the clarification node.

    Args:
        task_description: The original user task description
        task_rationale: Rationale from the planning node explaining why clarification is needed
    Returns:
        The constructed user prompt string
    """
    return (
        f"User Request: {task_description}\n\n"
        f"The task rationale is as follows:\n{task_rationale}\n\n"
        "Based on the above, please provide a list of clarification questions "
        "that need to be answered to proceed with the task."
    )
