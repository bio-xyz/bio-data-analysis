This project must implement code interpreter + data science agent.
Core functionality is split in two things:

1. Code interpreter that can execute code, read/write files, generate plots, etc. -> Using e2b_code_interpreter
2. Data science agent that can use the code interpreter to answer questions about data, generate reports, etc.

### Code interpreter

The code interpreter should be implemented using the e2b_code_interpreter package.
Basic code is:

```python
with Sandbox.create() as sbx:
        execution = sbx.run_code("print('hello world')")
        print(execution.logs)

        files = sbx.files.list("/")
        print(files)
```

### Data science agent

Agent has multiple steps:

1. Understand the user request -> make some overall plan
2. Break down the plan into smaller steps
3. For each step, generate code to be executed in the code interpreter
4. Execute the code in the code interpreter
5. Collect the results, analyze them, generate plots, etc.

User request consists of:

- Task description
- Data files (CSV, Excel, etc.) (Optional)
- Data files description (Optional)

In case of failure, agent has loop to fix the code and re-execute.

Agent should be built using LangChain framework.

The idle state q0 denotes
either task completion or readiness for new
instructions. State transitions are driven by
internally generated action signals and external
inputs, including user instructions and environment
feedback (i.e., execution success or failure). At
each state, DatawiseAgent produces two outputs:
an action, uniformly represented as markdown
and executable code cells, and an action signal
indicating the intended next state. The action is
executed in the notebook environment, yielding
external feedback. The agent determines its next
state via the transition function δ(q, σ, f), which
takes as input the current state q, the generated
action signal σ, and the feedback f from the
environment or user.

Ideal agent loop should be like this:

Algorithm 1 FST-based Multi-Stage Architecture
Require: I: task input, H: context history, AgentP : LLM
agent with language model P
1: Initialize context: H ← environment info and tools
2: H.update(I), q ← q0, σ ← I, f ← no_error
3: while True do
4: Generate action and action signal: A, signal ←
AgentP (q, H)
5: Execute action A and receive feedback f ∈
{error, no_error}
6: Determine the next state: q ← δ(q, σ, f)
7: Update context with executed A: H.update(A, f)
8: Update action signal: σ ← signal
9: if q = q0 then
10: Exit loop (Task complete or waiting for new instructions)
11: end if
12: end while
13: return H

### Combined agent + code interpreter

Agent is able to create new IPYNB Jupyter Notebook cells with markdown as description, small goal and then below that we have code.
That notebook should be copied to the sandbox, executed, and then pulled back into the agent for validation.
We should always execute whole notebook from start to end in the sandbox to ensure E2E functionality.

Notebook should be managed using nbformat package.

Final agent response should be a Jupyter Notebook with all the steps, plots, analysis, etc.
