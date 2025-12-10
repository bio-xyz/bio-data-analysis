"""Agent node implementations for the data science agent.

This module implements the FST-based multi-stage architecture with the following nodes:
- PLANNING_NODE: Entry point, decides between code execution or direct answer
- CODE_PLANNING_NODE: Plans and manages step-by-step code execution
- CODE_GENERATION_NODE: Generates code for current step
- CODE_EXECUTION_NODE: Executes code in sandbox
- ANSWERING_NODE: Generates final response
"""

import json
from pathlib import Path
from textwrap import indent

from app.agent.signals import ActionSignal
from app.agent.state import AgentState
from app.config import get_logger, settings
from app.models.structured_outputs import (
    ArtifactDecision,
    CodePlanningDecision,
    PlanningDecision,
    PythonCode,
)
from app.models.task import CompletedStep, TaskResponse
from app.services.executor_service import ExecutorService
from app.services.llm.llm_service import LLMService
from app.utils.nb_builder import NotebookBuilder
from app.utils.string_utils import truncate_output

logger = get_logger(__name__)


def planning_node(state: AgentState) -> dict:
    """
    PLANNING_NODE: Entry point for the agent flow.

    Analyzes the user request and decides whether to:
    - CODE_PLANNING: Task requires code execution
    - ANSWERING: Task can be answered directly or needs clarification

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with task_rationale and action_signal
    """
    logger.info("=== PLANNING_NODE ===")
    task_description = state.get("task_description", "")
    data_files_description = state.get("data_files_description", "")
    uploaded_files = state.get("uploaded_files", [])
    logger.info(f"Task: {task_description}")

    llm_service = LLMService(settings.PLANNING_LLM)

    decision: PlanningDecision = llm_service.generate_planning_decision(
        task_description=task_description,
        data_files_description=data_files_description,
        uploaded_files=uploaded_files,
    )

    logger.info(f"Planning decision: {decision.signal}")

    # Convert signal string to ActionSignal enum
    action_signal = ActionSignal.from_string(
        decision.signal, ActionSignal.CLARIFICATION
    )

    return {
        "task_rationale": decision.rationale,
        "action_signal": action_signal,
    }


def code_planning_node(state: AgentState) -> dict:
    """
    CODE_PLANNING_NODE: Main planning node for step-by-step code execution.

    Decides the next action:
    - ITERATE_CURRENT_STEP: Generate/regenerate code for current step
    - PROCEED_TO_NEXT_STEP: Move to next step after success
    - TASK_COMPLETE: Task complete successfully
    - TASK_FAILED: Abort due to unrecoverable error

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with current_step_goal, step management fields
    """
    # Input task context
    task_description = state.get("task_description")
    data_files_description = state.get("data_files_description")
    uploaded_files = state.get("uploaded_files")

    # Step context
    current_step_goal = state.get("current_step_goal", "")
    current_step_description = state.get("current_step_description", "")
    current_step_goal_history = state.get("current_step_goal_history", [])

    step_number = state.get("step_number", 0)
    step_attempts = state.get("step_attempts", 0)

    # Execution context
    generated_code = state.get("generated_code", "")
    last_execution_output = state.get("last_execution_output", "")
    last_execution_error = state.get("last_execution_error", None)
    execution_result = state.get("execution_result", None)

    # Output task context
    completed_steps = state.get("completed_steps", [])

    logger.info("=== CODE_PLANNING_NODE ===")
    logger.info(f"Step number: {step_number}")
    logger.info(f"Step attempts: {step_attempts}")
    logger.info(f"Completed steps: {len(completed_steps)}")

    if step_attempts > settings.CODE_PLANNING_MAX_STEP_RETRIES:
        logger.warning("Exceeded maximum step attempts, marking task as failed")
        return {
            "action_signal": ActionSignal.TASK_FAILED,
            "failure_reason": f"Exceeded maximum attempts for {current_step_goal}. Try simplifying the task or breaking it into smaller steps.",
        }

    # Decide next action
    llm_service = LLMService(settings.CODE_PLANNING_LLM)
    decision: CodePlanningDecision = llm_service.generate_code_planning_decision(
        task_description=task_description,
        data_files_description=data_files_description,
        uploaded_files=uploaded_files,
        current_step_goal=current_step_goal,
        current_step_goal_history=current_step_goal_history,
        last_execution_output=last_execution_output,
        last_execution_error=last_execution_error,
        completed_steps=completed_steps,
    )

    new_action_signal = ActionSignal.from_string(
        decision.signal, ActionSignal.ITERATE_CURRENT_STEP
    )
    new_step_goal = decision.step_goal or current_step_goal
    new_step_description = decision.step_description
    reasoning = decision.reasoning
    observations = decision.observations

    obs_json = json.dumps([obs.model_dump() for obs in observations], indent=2)
    logger.info(f"Observations:\n{obs_json}")

    # Update world state
    updates = {
        # Reset code generation attempts
        "code_generation_attempts": 0,
        # Reset execution context
        "generated_code": "",
        "execution_result": None,
        "last_execution_error": None,
        "last_execution_output": "",
    }

    # Separate decision branches
    if new_action_signal == ActionSignal.ITERATE_CURRENT_STEP:
        logger.info("Decided to ITERATE_CURRENT_STEP")
        updates.update(
            {
                # Action
                "action_signal": new_action_signal,
                # Step configuration
                "current_step_goal": new_step_goal,
                "current_step_description": new_step_description,
                "current_step_goal_history": current_step_goal_history
                + [current_step_goal],
                "step_attempts": step_attempts + 1,
            }
        )
    elif new_action_signal == ActionSignal.PROCEED_TO_NEXT_STEP:
        logger.info("Decided to PROCEED_TO_NEXT_STEP")
        updates.update(
            {
                # Action
                "action_signal": new_action_signal,
                # Step configuration
                "step_number": step_number + 1,
                "current_step_goal": new_step_goal,
                "current_step_description": new_step_description,
                "current_step_goal_history": [new_step_goal],
                "step_attempts": 0,
            }
        )
    elif new_action_signal in (ActionSignal.TASK_COMPLETED, ActionSignal.TASK_FAILED):
        logger.info(f"Decided to {new_action_signal.name}")
        updates["action_signal"] = new_action_signal
        if new_action_signal == ActionSignal.TASK_FAILED:
            updates["failure_reason"] = reasoning
    else:
        raise ValueError(f"Unknown decision signal: {decision.signal}")

    if new_action_signal in (
        ActionSignal.PROCEED_TO_NEXT_STEP,
        ActionSignal.TASK_COMPLETED,
        ActionSignal.TASK_FAILED,
    ):
        logger.info("Finalizing current step before proceeding")
        completed_step = CompletedStep(
            step_number=step_number,
            goal=current_step_goal,
            description=current_step_description,
            code=generated_code,
            execution_result=execution_result,
            success=not last_execution_error,
            observations=observations,
        )
        updates["completed_steps"] = completed_steps + [completed_step]

    return updates


def code_generation_node(state: AgentState) -> dict:
    """
    CODE_GENERATION_NODE: Generates code for the current step.

    Takes the current step goal and generates Python code to execute.

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with generated_code
    """
    # Input context
    data_files_description = state.get("data_files_description")
    uploaded_files = state.get("uploaded_files")

    # Step context
    current_step_goal = state.get("current_step_goal", "")
    current_step_description = state.get("current_step_description", "")
    code_generation_attempts = state.get("code_generation_attempts", 0)

    # Previous attempt context
    last_execution_output = state.get("last_execution_output", "")
    last_execution_error = state.get("last_execution_error", None)
    previous_code = state.get("generated_code", "")

    # Notebook code
    completed_steps = state.get("completed_steps", [])

    logger.info("=== CODE_GENERATION_NODE ===")
    logger.info(f"Generating code for step: {current_step_goal}")

    llm_service = LLMService(settings.CODE_GENERATION_LLM)

    # Get previous code from completed steps for context
    notebook_code = ""
    if completed_steps:
        for step in completed_steps:
            notebook_code += f"\n\n# Step {step.step_number}: {step.goal}\n{step.code}"

    code_result: PythonCode = llm_service.generate_step_code(
        data_files_description=data_files_description,
        uploaded_files=uploaded_files,
        current_step_goal=current_step_goal,
        current_step_description=current_step_description,
        last_execution_output=last_execution_output,
        last_execution_error=last_execution_error,
        notebook_code=notebook_code,
        previous_code=previous_code,
    )

    logger.info(f"Generated code length: {len(code_result.code)} characters")
    logger.info(f"Code generated: {code_result.code}...")

    return {
        "generated_code": code_result.code,
        "code_generation_attempts": code_generation_attempts + 1,
        "action_signal": ActionSignal.EXECUTE_CODE,
    }


def code_execution_node(state: AgentState) -> dict:
    """
    CODE_EXECUTION_NODE: Executes generated code in the sandbox.

    Runs the code and captures output/errors.

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with execution results
    """
    current_step_goal = state.get("current_step_goal", "")
    sandbox_id = state.get("sandbox_id", None)
    generated_code = state.get("generated_code", "")

    logger.info("=== CODE_EXECUTION_NODE ===")
    logger.info(f"Executing code for step: {current_step_goal}")

    executor = ExecutorService()

    try:
        execution = executor.execute_code(sandbox_id, generated_code)

        if execution.error:
            error_msg = str(execution.logs.stderr or execution.error)
            logger.warning(f"Code execution failed: {error_msg}")

            return {
                "execution_result": execution,
                "last_execution_error": truncate_output(error_msg),
                "last_execution_output": truncate_output(execution.logs.stdout),
                "action_signal": ActionSignal.CODE_EXECUTION_FAILED,
            }

        output = ""
        if execution.logs.stdout:
            output += "\n[stdout]\n" + truncate_output("\n".join(execution.logs.stdout))
        if execution.results:
            results = [str(r) for r in execution.results]
            output += "\n[results]\n" + truncate_output("\n".join(results))

        logger.info("Code execution succeeded")
        logger.info(f"Execution output: {output}...")

        return {
            "execution_result": execution,
            "last_execution_output": output or "(no output)",
            "last_execution_error": "",
            "action_signal": ActionSignal.CODE_EXECUTION_SUCCESS,
        }

    except Exception as e:
        logger.error(f"Execution exception: {e}")
        return {
            "last_execution_error": truncate_output(str(e)),
            "last_execution_output": "",
            "action_signal": ActionSignal.CODE_EXECUTION_FAILED,
        }


def answering_node(state: AgentState) -> dict:
    """
    ANSWERING_NODE: Generates the final response.

    Creates the final task response based on:
    - Completed steps and their results
    - Any errors or failures
    - Clarification questions if needed

    Args:
        state: Current agent state

    Returns:
        dict: Updated state with task_response
    """

    action_signal = state.get("action_signal")
    task_description = state.get("task_description", "")
    task_rationale = state.get("task_rationale", "")

    llm_service = LLMService(settings.ANSWERING_LLM)

    if action_signal == ActionSignal.CLARIFICATION:
        clarification_response = llm_service.generate_clarification_questions(
            task_description=task_description,
            task_rationale=task_rationale,
        )
        return {
            "task_response": TaskResponse(
                answer=clarification_response.questions, success=False
            ),
            "action_signal": ActionSignal.FINAL_ANSWER,
        }
    elif action_signal == ActionSignal.GENERAL_ANSWER:
        general_answer_response = llm_service.generate_general_answer(
            task_description=task_description,
            task_rationale=task_rationale,
        )
        return {
            "task_response": TaskResponse(
                answer=general_answer_response.answer, success=True
            ),
            "action_signal": ActionSignal.FINAL_ANSWER,
        }
    elif action_signal not in (
        ActionSignal.TASK_COMPLETED,
        ActionSignal.TASK_FAILED,
    ):
        logger.error(
            f"ANSWERING_NODE reached with invalid action signal: {action_signal}"
        )
        return {
            "task_response": TaskResponse(
                answer="Error: ANSWERING_NODE reached with invalid action signal.",
                success=False,
            ),
            "action_signal": ActionSignal.FINAL_ANSWER,
        }

    sandbox_id = state.get("sandbox_id", None)
    completed_steps = state.get("completed_steps", [])
    failure_reason = state.get("failure_reason", None)
    execution_result = state.get("execution_result", None)

    logger.info("=== ANSWERING_NODE ===")
    logger.info("Generating final response")
    logger.info(f"Completed steps: {len(completed_steps)}")

    executor_service = ExecutorService()
    workdir_contents = executor_service.print_limited_tree(sandbox_id)

    task_answer = llm_service.generate_task_response_answer(
        task_description=task_description,
        completed_steps=completed_steps,
        failure_reason=failure_reason,
        workdir_contents=workdir_contents,
    )

    # Fix artifact paths to be absolute
    for artifact in task_answer.artifacts:
        path = Path(artifact.full_path)
        full_path = (
            path
            if path.is_absolute()
            else Path(settings.DEFAULT_WORKING_DIRECTORY) / path
        )
        artifact.full_path = str(full_path)

    # Build and save notebook from completed steps
    nb_builder = NotebookBuilder()

    # Add introduction markdown
    quoted_task = indent(task_description, "> ")
    nb_builder.add_markdown(
        f"# Task\n\n{quoted_task}\n\n## Rationale\n\n{task_rationale}"
    )

    # Add completed steps
    for step in completed_steps:
        nb_builder.add_markdown(f"## Step {step.step_number}: {step.goal}")
        nb_builder.add_markdown(step.description)

        if step.code:
            nb_builder.add_code(step.code)

        if step.execution_result:
            nb_builder.add_execution(step.execution_result)

    # Save notebook to sandbox
    notebook_path = executor_service.save_notebook_to_sandbox(
        sandbox_id, nb_builder.build()
    )

    notebook_description = (
        task_answer.notebook_description
        or "Jupyter notebook documenting the task execution steps."
    )

    task_answer.artifacts.append(
        ArtifactDecision(
            type="FILE",
            description=notebook_description,
            full_path=notebook_path,
        )
    )

    logger.info("Final response generated")

    return {
        "task_answer": task_answer,
        "action_signal": ActionSignal.FINAL_ANSWER,
    }
