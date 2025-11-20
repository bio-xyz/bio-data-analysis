from typing import Any, Dict, List, Optional

import nbformat
from e2b_code_interpreter.models import Execution, Result


class NotebookBuilder:
    """
    Builder class to create and manage an ordered list of notebook cells.

    Usage:
        builder = NotebookBuilder()
        builder.add_code("print('Hello')")
        builder.add_markdown("# Title")
        builder.add_code("x = 42")

        # Add execution results
        builder.add_output(result)

        # Save or build
        notebook = builder.build()
        builder.save("output.ipynb")
    """

    def __init__(self):
        """Initialize an empty notebook builder."""
        self.cells: List[Dict[str, Any]] = []
        self._code_execution_count = 1

    def add_code(
        self, content: str, execution_count: Optional[int] = None
    ) -> "NotebookBuilder":
        """
        Add a code cell to the notebook.

        Args:
            content: The Python code to add.
            execution_count: Optional execution count. If not provided, auto-increments.

        Returns:
            Self for method chaining.
        """
        if execution_count is None:
            execution_count = self._code_execution_count
            self._code_execution_count += 1

        cell = nbformat.v4.new_code_cell(content, execution_count=execution_count)
        self.cells.append(cell)
        return self

    def add_markdown(self, content: str) -> "NotebookBuilder":
        """
        Add a markdown cell to the notebook.

        Args:
            content: The markdown text to add.

        Returns:
            Self for method chaining.
        """
        cell = nbformat.v4.new_markdown_cell(content)
        self.cells.append(cell)
        return self

    def _ensure_last_cell_is_code(self) -> Dict[str, Any]:
        """
        Validate that the last cell is a code cell and return it.

        Returns:
            The last code cell.

        Raises:
            ValueError: If there are no cells or the last cell is not a code cell.
        """
        if not self.cells:
            raise ValueError("Cannot add output: no cells in the notebook")

        last_cell = self.cells[-1]
        if last_cell.get("cell_type") != "code":
            raise ValueError("Cannot add output: last cell is not a code cell")

        # Initialize outputs list if not present
        if "outputs" not in last_cell:
            last_cell["outputs"] = []

        return last_cell

    def add_output(
        self, result: Result, execution_count: Optional[int] = None
    ) -> "NotebookBuilder":
        """
        Add an output to the last code cell from an e2b_code_interpreter Result.

        Args:
            result: The Result object to convert and add as output.
            execution_count: The execution count to include in the output (only for execute_result).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If there are no cells or the last cell is not a code cell.
        """
        last_cell = self._ensure_last_cell_is_code()

        # Convert Result to notebook output
        data = {}

        # Map Result fields to MIME types
        if result.text:
            data["text/plain"] = result.text
        if result.html:
            data["text/html"] = result.html
        if result.markdown:
            data["text/markdown"] = result.markdown
        if result.svg:
            data["image/svg+xml"] = result.svg
        if result.png:
            data["image/png"] = result.png
        if result.jpeg:
            data["image/jpeg"] = result.jpeg
        if result.pdf:
            data["application/pdf"] = result.pdf
        if result.latex:
            data["text/latex"] = result.latex
        if result.json:
            data["application/json"] = result.json
        if result.javascript:
            data["application/javascript"] = result.javascript

        # Handle extra data - assuming keys are MIME types
        if result.extra:
            for key, value in result.extra.items():
                data[key] = value

        output_type = "execute_result" if result.is_main_result else "display_data"

        kwargs = {}
        if execution_count and output_type == "execute_result":
            kwargs["execution_count"] = execution_count

        output = nbformat.v4.new_output(output_type, data=data, **kwargs)
        last_cell["outputs"].append(output)
        return self

    def add_execution(self, execution: Execution) -> "NotebookBuilder":
        """
        Add all outputs from an e2b Execution to the last code cell.

        This method processes the complete Execution object including:
        - stdout/stderr logs (each line as a separate output)
        - execution results (display outputs and main result)
        - errors

        Args:
            execution: The Execution object from e2b code interpreter.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If there are no cells or the last cell is not a code cell.
        """
        last_cell = self._ensure_last_cell_is_code()

        # Add stdout logs as separate stream outputs
        for stdout_line in execution.logs.stdout:
            if stdout_line:
                stdout_output = nbformat.v4.new_output(
                    "stream", name="stdout", text=stdout_line
                )
                last_cell["outputs"].append(stdout_output)

        # Add stderr logs as separate stream outputs
        for stderr_line in execution.logs.stderr:
            if stderr_line:
                stderr_output = nbformat.v4.new_output(
                    "stream", name="stderr", text=stderr_line
                )
                last_cell["outputs"].append(stderr_output)

        # Add all results (display data and execute_result) by calling add_output
        for result in execution.results:
            self.add_output(result, execution.execution_count)

        # Add error output if present
        if execution.error:
            error_output = nbformat.v4.new_output(
                "error",
                ename=execution.error.name,
                evalue=execution.error.value,
                traceback=execution.error.traceback.split("\n"),
            )
            last_cell["outputs"].append(error_output)

        # Update execution count on the cell if available
        if execution.execution_count:
            last_cell["execution_count"] = execution.execution_count

        return self

    def build(self, metadata: Optional[Dict[str, Any]] = None) -> nbformat.NotebookNode:
        """
        Build the complete notebook structure.

        Args:
            metadata: Optional metadata for the notebook.

        Returns:
            A complete nbformat NotebookNode.
        """
        if metadata is None:
            metadata = {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0",
                },
            }

        return nbformat.v4.new_notebook(cells=self.cells, metadata=metadata)

    def clear(self) -> "NotebookBuilder":
        """
        Clear all cells from the builder.

        Returns:
            Self for method chaining.
        """
        self.cells = []
        self._code_execution_count = 1
        return self
