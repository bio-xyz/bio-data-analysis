"""Prompts package -  FST-based multi-stage architecture."""

from app.prompts.clarification import (
    build_task_clarification_prompt,
    get_task_clarification_system_prompt,
)
from app.prompts.code_generation import (
    build_code_generation_prompt,
    get_code_generation_system_prompt,
)
from app.prompts.code_planning import (
    build_code_planning_prompt,
    get_code_planning_system_prompt,
)
from app.prompts.general_answer import (
    build_general_answer_prompt,
    get_general_answer_system_prompt,
)
from app.prompts.planning import build_planning_prompt, get_planning_system_prompt
from app.prompts.task_response_generation import (
    build_task_response_prompt,
    get_task_response_system_prompt,
)

__all__ = [
    # Code generation
    "build_code_generation_prompt",
    "get_code_generation_system_prompt",
    # Code planning ()
    "build_code_planning_prompt",
    "get_code_planning_system_prompt",
    # Planning ()
    "build_planning_prompt",
    "get_planning_system_prompt",
    # Task response
    "build_task_response_prompt",
    "get_task_response_system_prompt",
    # Clarification
    "build_task_clarification_prompt",
    "get_task_clarification_system_prompt",
    # General answering
    "get_general_answer_system_prompt",
    "build_general_answer_prompt",
]
