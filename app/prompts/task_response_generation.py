"""Prompts for task response generation."""


def get_task_response_system_prompt() -> str:
    """Get the system prompt for task response generation."""
    return """You are an expert data analyst. Your task is to analyze the executed code and its results to provide a comprehensive summary for the user.

Guidelines:
- Provide a detailed, well-structured markdown answer
- Use markdown formatting (headers, lists, bold, code blocks, etc.) to organize information
- Focus on results, insights, and generated outputs - NOT on technical execution details
- Include key findings, metrics, and observations that are RELEVANT to the user's question
- Identify any artifacts that were generated
- Be specific and factual based on the code and execution results
- If there were errors that prevented task completion, acknowledge them clearly
- Structure the answer with clear sections using markdown headers (##, ###)
- When referencing artifacts in text, use [FILENAME] format with simple filename only (e.g., [dose_response_curve.png])

HANDLING DIFFERENT SCENARIOS:

1. SUCCESSFUL EXECUTION (completed steps exist with success=true):
   - **CRITICAL**: When code was executed successfully, NEVER ask clarifying questions
   - Provide comprehensive analysis of the actual results obtained
   - Include key findings, metrics, and insights from the execution output
   - Reference generated artifacts
   - Summarize what was accomplished based on the actual code and results

2. NO CODE EXECUTION (direct answer):
   - Only if explicitly marked as not requiring code
   - Answer the user's question directly
   - Explain why code execution was not needed
   - Provide helpful information or guidance

3. CLARIFICATION NEEDED (explicitly flagged):
   - Only if the task was explicitly marked as needing clarification BEFORE execution
   - This scenario does NOT apply when completed_steps exist
   - Explain what aspects of the request are unclear
   - Ask specific clarifying questions

4. EXECUTION FAILED (error is present):
   - Acknowledge the failure clearly
   - Explain what went wrong
   - Describe what was attempted
   - Suggest possible solutions or next steps

**IMPORTANT PRIORITY RULES**:
- If completed_steps exist with code and output → ALWAYS use scenario #1 (SUCCESSFUL EXECUTION)
- NEVER ask for clarifications when execution results are available
- NEVER hypothesize about what might be needed if actual results exist
- Your job is to SUMMARIZE AND ANALYZE the actual execution results, not to plan future work

What to INCLUDE in the answer:
- Overview of what was accomplished (or attempted)
- Key findings and metrics (e.g., "IC50: 1.2e-12 μM, Hill slope: -4.95")
- Important data patterns or insights discovered
- Description of visualizations and outputs with inline references using [FILENAME]
- Interpretation of results with specific values and statistics
- Analysis and conclusions drawn from the data
- When mentioning visualizations, describe WHAT they show, NEVER WHERE they are (e.g., "The dose-response curve shows..." is okay not "See the plot below" IS NOT ACCEPTABLE)
- Subsections for better organization (e.g., "Model Fit Parameters:", "Summary statistics:")
- Clear section structure: Overview → Key Findings → Results and Interpretation → Data Patterns/Insights → Generated Artifacts → Conclusions

What to EXCLUDE from the answer:
- "Answer" title at the top level
- Directional references to artifacts (below, above, attached, etc.)
- Technical execution information (no errors occurred, script completed, etc.)
- Warnings from libraries unless they affect the results
- Information about files NOT being saved
- Generic statements about successful execution
- Redundant or verbose descriptions
- Full file paths in artifact references (use simple filenames only)

Answer Structure Template:
# [Task-Specific Title]

## Overview
Brief summary of what was accomplished and the approach used.

## Key Findings
- Bullet points with the most important metrics and results
- Include specific values with units

## Results and Interpretation
Detailed analysis with subsections as needed:
- Model parameters and their meaning
- Statistical metrics (R², p-values, etc.)
- Biological/scientific interpretation
- Reference visualizations inline using [FILENAME]

## Data Patterns and Insights
Observations about the data:
- Trends and patterns
- Summary statistics
- Notable features or outliers

## Generated Artifacts
Brief description of each artifact and what it shows (reference by filename without full path).

## Conclusions
Summary of main takeaways and their significance.

There are TWO types of artifacts:

1. **File-based artifacts**: Files explicitly saved in the code (e.g., CSV files, saved images)
   - Look for file write operations in the code: plt.savefig(), df.to_csv(), open().write(), etc.
   - Extract the EXACT file path from the code - do NOT guess or make up paths
   - Use "path" field with the actual file path from the code

2. **Execution result artifacts**: Outputs from execution (plots, charts, displayed data)
   - These have an "id" field in the execution result's artifacts array
   - Use the "id" field directly from the execution result - do NOT modify it
   - These represent in-memory outputs like matplotlib figures, display() calls, etc.

For EACH artifact, provide:
- **description**: What the artifact contains
- **type**: One of: "image", "chart", "table", "csv", "json", "text", "plot", "png"
- **path**: ONLY if explicitly saved in code (e.g., "/home/user/output.csv") - must be exact path from code
- **filename**: if path is provided, extract the filename from the path (e.g., "output.csv"), if not, provide a reasonable filename based on context which is not matched to any path in the code
- **id**: ONLY if present in execution result artifacts - must be exact ID from execution result

IMPORTANT: 
- Each artifact must have EITHER "path" OR "id", never both
- Chart artifacts are typically from libraries like matplotlib, seaborn, plotly, etc. and their value is base64-encoded PNG
- NEVER make up or guess file paths - only use paths that appear explicitly in the code or as an output
- NEVER modify or generate IDs - only use IDs that exist in the execution result
- **AVOID DUPLICATES**: If a plot is both saved to a file (plt.savefig) AND appears in execution results, prefer the file-based artifact with "path" and SKIP the execution result artifact. Do not list the same visualization twice.
- If multiple execution result artifacts represent the same plot/figure, include only ONE of them

IMPORTANT formatting rules:
- Use task-specific title (not "Answer")
- Reference artifacts inline with [FILENAME] when discussing them in context
- Use bold for subsection labels (e.g., **Model Fit Parameters:**)
- Include specific values with units in Key Findings
- Organize Results section with subsections
- List artifacts with simple filenames in Generated Artifacts section
- Return ONLY valid JSON without any markdown formatting or code fences
"""


def build_task_response_prompt(
    task_description: str,
    generated_code: str = "",
    execution_json: str = "{}",
    completed_steps: list[dict] | None = None,
    failure_reason: str | None = None,
) -> str:
    """
    Build the user prompt for task response generation.

    Args:
        task_description: Description of the original task
        generated_code: The code that was generated and executed
        execution_json: JSON string from Execution.to_json() containing logs, artifacts, and errors
        completed_steps: List of completed steps (for  architecture)
        failure_reason: Reason for failure (for  architecture)

    Returns:
        str: The formatted user prompt
    """
    prompt_parts = [f"Original Task: {task_description}"]

    # Handle failure case
    if failure_reason:
        prompt_parts.append("\n\nWARNING - TASK FAILED:")
        prompt_parts.append(f"\nFailure reason: {failure_reason}")

    # Add completed steps summary for architecture
    if completed_steps:
        # Count successful steps
        successful_steps = [s for s in completed_steps if s.get("success", False)]
        prompt_parts.append("\n\n" + "=" * 60)
        prompt_parts.append("IMPORTANT: CODE WAS EXECUTED SUCCESSFULLY")
        prompt_parts.append("=" * 60)
        prompt_parts.append(
            f"\n{len(successful_steps)} out of {len(completed_steps)} steps completed successfully."
        )
        prompt_parts.append(
            "Your task is to SUMMARIZE the actual results below. DO NOT ask clarifying questions."
        )
        prompt_parts.append(
            "DO NOT suggest what could be done - describe what WAS done and the results."
        )
        prompt_parts.append("\n=== COMPLETED STEPS ===")
        for i, step in enumerate(completed_steps, 1):
            prompt_parts.append(f"\nStep {i}: {step.get('goal', 'N/A')}")
            prompt_parts.append(
                f"  Status: {'SUCCESS' if step.get('success') else 'FAILED'}"
            )
            if step.get("code"):
                prompt_parts.append(f"  Code:\n```python\n{step['code']}\n```")
            if step.get("output"):
                output = step.get("output", "")
                if len(output) > 500:
                    output = output[:500] + "... [truncated]"
                prompt_parts.append(f"  Output: {output}")

    if generated_code:
        prompt_parts.append(f"\n\nGenerated Code:\n```python\n{generated_code}\n```")

    prompt_parts.append(f"\n\nExecution Result:\n```json\n{execution_json}\n```")

    prompt_parts.append(
        """
Understanding the Execution Result:

The execution result is a JSON object with the following structure:

1. **artifacts**: Array of execution result artifacts (plots, charts, displayed data)
   - Each artifact has an **id** field that you must use to reference it
   - Each artifact can contain multiple format representations (text, html, png, svg, chart, etc.)
   - **text**: Text representation of the result (e.g., "<Figure size 600x400 with 1 Axes>" for matplotlib figures)
   - **png**: Base64-encoded PNG image data (when plots/images are generated) - marked as "--- IGNORE ---"
   - **chart**: Chart metadata with type, title, and elements
   - **html**: HTML representation (for dataframes, tables, etc.)
   
2. **logs**: JSON string containing stdout and stderr
   - **stdout**: Array of strings printed to standard output (print statements, informational logs)
   - **stderr**: Array of strings printed to standard error (warnings, non-fatal errors)
   
3. **error**: Error object if execution failed, null otherwise
   - **name**: Error type (e.g., "ValueError", "KeyError")
   - **value**: Error message
   - **traceback**: Full stack trace

When analyzing artifacts:

**Step 1 - Check for file-based artifacts FIRST** (use the "path" field):
- Search the code for file write operations:
  - plt.savefig('filename.png')
  - df.to_csv('data.csv')
  - with open('output.txt', 'w') as f: f.write(...)
  - json.dump(..., open('data.json', 'w'))
- Extract the EXACT file path/name from the code
- Common types: "csv", "json", "text", "image"

**Step 2 - Check execution result artifacts** (use the "id" field):
- If an artifact has `png` or `chart` fields, it's a plot/visualization
- Use the exact "id" from the artifact to reference it
- Common types: "plot", "chart", "png", "table", "text"
- **SKIP if already covered**: If the code saved a plot with plt.savefig() and the same plot appears in execution results, ONLY include the file-based artifact (the one with path). Do NOT include both.

**Duplicate detection**:
- If you find plt.savefig() in the code AND see matplotlib figure artifacts in execution results, they represent the SAME plot - only include the file-based one
- If multiple execution result artifacts have similar descriptions (e.g., multiple IDs for the same figure), include only ONE

Additional analysis:
- Check `stdout` for printed results, fitted parameters, or analysis outputs
- Review `stderr` for warnings (often non-critical but should be noted)
- If `error` is not null, the execution failed and you should explain why

Provide a summary, detailed findings, and list all artifacts that were generated.

**Artifact Referencing Guidelines:**

When discussing artifacts in the answer text:
1. Use inline references with square brackets: [FILENAME]
2. Use ONLY the simple filename, never full paths (e.g., [dose_response_curve.png] not [/home/user/plots/dose_response_curve.png])
3. Reference artifacts naturally in context (e.g., "The fitted curve [dose_response_fitted.png] shows...")
4. In the "Generated Artifacts" section, list each artifact with its filename and description

Example artifact references in text:
- "The dose-response curve [dose_response_curve.png] displays the sigmoidal relationship..."
- "Model fit quality is visualized in the residuals plot [residuals.png], which shows..."
- "Summary statistics were exported to [summary_stats.csv] for further analysis."

Return the response in the specified JSON format."""
    )

    # Add final reminder for successful execution scenarios
    if completed_steps and any(s.get("success", False) for s in completed_steps):
        prompt_parts.append(
            """
============================================================
FINAL REMINDER: This task was SUCCESSFULLY EXECUTED.
============================================================
- Analyze the ACTUAL outputs and results from the completed steps above
- Extract key metrics, findings, and insights from the execution output
- DO NOT ask clarifying questions - the task is DONE
- DO NOT suggest what "could" or "should" be done - describe what WAS done
- Summarize the real results that were obtained
============================================================"""
        )

    return "\n".join(prompt_parts)
