"""Prompts for code generation."""

from typing import Optional


def get_code_generation_system_prompt() -> str:
    """Get the system prompt for code generation."""
    return """You are an expert data science code generator working in a Jupyter notebook environment.

You are code generation agent responsible for writing Python code to accomplish specific data science tasks in a Jupyter notebook.
You are being overseen by a planning agent that provides you with step goals and feedback based on code execution results.
Your job is to write the code for the NEXT cell in the notebook based on the current step goal and context.

## CORE PRINCIPLE: CONTINUATION, NOT REPETITION
You are writing the NEXT cell in an already-running notebook. All previous cells have been executed.
All variables, imports, and data from previous cells ARE ALREADY IN MEMORY and available to use directly.

## STRICT RULES

### 0. CONTEXT AWARENESS
- ALWAYS consider the code that has already been executed in previous cells
- Build upon existing variables, DataFrames, and imports
- DO NOT re-import, re-define, or re-load anything that was done in previous cells
- Your code MUST integrate seamlessly with the existing notebook state
- Your code is allowed to fail. You are being overseen by a planning agent that will guide you based on execution results.

### 1. NEVER RE-IMPORT OR RE-DEFINE
- DO NOT import libraries that were imported in previous cells
- DO NOT reload data files that were already loaded
- DO NOT redefine variables, DataFrames, or functions from previous cells
- ASSUME all previous variables exist and are ready to use

### 2. VARIABLE CONTINUITY
- Use the EXACT variable names from previous cells (e.g., if data was loaded as `df`, use `df`)
- Build upon existing DataFrames and variables, don't recreate them
- If you need a modified version, create a NEW variable name (e.g., `df_filtered`, `df_cleaned`)

### 3. PACKAGE INSTALLATION
- If a package needs installation, use: !pip install package_name
- NEVER use subprocess, os.system(), or any other method to install packages
- Only install if the package is genuinely missing and not a standard data science library

### 4. CODE STYLE
- Write minimal, focused code for the CURRENT STEP only
- You DO NOT need to write try/except blocks unless needed for error handling, you are allowed to fail
- Add brief comments only where logic is non-obvious
- Print key results/outputs for visibility
- Displayed data and printed outputs should be concise and relevant; main agent will always truncate long outputs, so some key results might not be visible
- Use plt.show() after plots (don't save unless specifically requested)

### 5. ERROR RECOVERY
When fixing a failed attempt:
- Analyze the error message carefully
- Fix the specific issue, don't rewrite everything
- Maintain variable names from the working previous cells
- Check if the error was due to a missing variable that should exist

## OUTPUT FORMAT
CRITICAL: Return ONLY valid JSON without any markdown formatting or code fences
"""


def build_code_generation_prompt(
    current_step_goal: str,
    current_step_description: Optional[str] = None,
    data_files_description: Optional[str] = None,
    uploaded_files: Optional[list[str]] = None,
    last_execution_output: Optional[str] = None,
    last_execution_error: Optional[str] = None,
    notebook_code: Optional[str] = None,
    previous_code: Optional[str] = None,
) -> str:
    """
    Build the user prompt for code generation.

    Args:
        current_step_goal (str): The goal of the current step
        current_step_description (Optional[str]): Optional detailed description of the current step
        data_files_description (Optional[str]): Optional description of data files
        uploaded_files (Optional[list[str]]): Optional list of uploaded file names
        last_execution_output (Optional[str]): Output from last execution
        last_execution_error (Optional[str]): Error from last execution
        notebook_code (Optional[str]): Code already present in the notebook
        previous_code (Optional[str]): Previously generated code for context

    Returns:
        str: The formatted user prompt
    """
    prompt_parts = []

    # Data files context
    if data_files_description:
        prompt_parts.append("## DATA FILES DESCRIPTION")
        prompt_parts.append(data_files_description)

    if uploaded_files:
        prompt_parts.append("## UPLOADED FILES")
        prompt_parts.append(
            "The following files have been uploaded and are available for use:"
        )
        for path in uploaded_files:
            prompt_parts.append(f"- {path}")
        prompt_parts.append(
            "Note: There can be multiple other fiels as well that were generated during code execution and are available in the notebook environment."
        )

    # Notebook state - what's already been done and what variables exist
    if notebook_code:
        prompt_parts.append("## NOTEBOOK STATE: Previous Code Executed")
        prompt_parts.append(
            "The following code has already been executed in previous cells:"
        )
        prompt_parts.append(f"```python\n{notebook_code}\n```")

    # Current step (the actual request)
    prompt_parts.append("## YOUR TASK: Write code for this step ONLY")
    prompt_parts.append(f"**Step Goal:** {current_step_goal}")

    if current_step_description:
        prompt_parts.append(f"**Details:** {current_step_description}")

    # Error context for retries
    if last_execution_error and previous_code:
        prompt_parts.append(f"\n## PREVIOUS ATTEMPT FAILED WITH ERROR")
        prompt_parts.append("Your previous code for the current step:")
        prompt_parts.append(f"```python\n{previous_code}\n```")
        prompt_parts.append(
            f"\n**Output from last execution:**\n{last_execution_output}"
        )
        prompt_parts.append(f"\n**Error:**\n{last_execution_error}")
        prompt_parts.append(
            "\n**Fix the error. Do NOT rewrite working code from previous cells.**"
        )
        prompt_parts.append(
            "**Identify what went wrong and make a minimal, targeted fix.**"
        )

    # Final instruction
    prompt_parts.append("\n---")
    prompt_parts.append(
        "Generate ONLY the Python code for this step. No markdown, no code fences. Code will be placed in the next cell of an existing Jupyter notebook."
    )
    if notebook_code:
        prompt_parts.append(
            "Remember: Previous cells already ran. Use existing variables directly!"
        )

    return "\n".join(prompt_parts)
