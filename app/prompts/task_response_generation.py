"""Prompts for task response generation."""


def get_task_response_system_prompt() -> str:
    """Get the system prompt for task response generation."""
    return """You are an expert data analyst. Your task is to analyze the executed code and its results to provide a comprehensive summary for the user.

Guidelines:
- Provide a clear, concise summary of what was accomplished
- List detailed findings and observations from the execution that are RELEVANT to the user's question
- Focus on results, insights, and generated outputs - NOT on technical execution details
- Identify any artifacts that were generated
- Be specific and factual based on the code and execution results
- If there were errors that prevented task completion, acknowledge them clearly

What to INCLUDE in details (keep it SHORT - 1-6 key points maximum):
- The most critical findings or key metrics (e.g., "IC50: 1.2e-12 μM, Hill slope: -4.95")
- Important data patterns or insights discovered
- What the main visualization or output represents (if not obvious from the summary)

What to EXCLUDE from details:
- Technical execution information (no errors occurred, script completed, etc.)
- Warnings from libraries unless they affect the results
- Information about files NOT being saved
- Generic statements about successful execution
- Verbose descriptions or step-by-step explanations
- Redundant information already in the summary

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

Return your response in the following JSON format:
{
  "summary": "A concise summary of the findings",
  "details": ["Detail 1", "Detail 2", "Detail 3"],
  "artifacts": [
    {
      "description": "Description of the artifact",
      "type": "png|chart|table|csv|json|text|plot",
      "path": "/exact/path/from/code.ext",
      "filename": "output.ext",
      "id": null
    },
    {
      "description": "Description of execution result artifact",
      "type": "png|chart|table|csv|json|text|plot",
      "path": null,
      "filename": "output.ext",
      "id": "uuid-from-execution-result"
    }
  ]
}

Return ONLY valid JSON without any markdown formatting or code fences.
"""


def build_task_response_prompt(
    task_description: str,
    generated_code: str,
    execution_json: str,
    success: bool = True,
    error: str | None = None,
) -> str:
    """
    Build the user prompt for task response generation.

    Args:
        task_description: Description of the original task
        generated_code: The code that was generated and executed
        execution_json: JSON string from Execution.to_json() containing logs, artifacts, and errors
        success: Flag indicating whether the operation was successful (plan or execution)
        error: Error message if planning or execution failed

    Returns:
        str: The formatted user prompt
    """
    prompt_parts = [f"Original Task: {task_description}"]

    if not success and error:
        prompt_parts.append(f"\n\nWARNING - ERROR OCCURRED:\n{error}")

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

Provide a summary, detailed findings, and list all artifacts that were generated. Return the response in the specified JSON format."""
    )

    return "\n".join(prompt_parts)
