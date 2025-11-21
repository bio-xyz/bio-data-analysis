# Bio Code Interpreter

A data science agent with integrated code interpreter for executing Python code, analyzing data, and generating insights.

## Features

- **Code Interpreter**: Execute Python code in isolated E2B sandboxes
- **Data Science Agent**: Multi-stage FST-based agent for data analysis workflows
- **LLM Support**: Compatible with OpenAI and Anthropic models
- **Automatic Retry Logic**: Intelligent error handling with code regeneration
- **Plan-Execute-Analyze**: Structured workflow for complex data tasks

## Agent Workflow

```
START
  ↓
[plan]  ───────────→  Generate execution plan
  ↓
  ├─ SUCCESS ────────→  [code_generation]  ──→  Generate Python code
  │                                                  ↓
  │                                            [execution]  ────────→  Execute code in sandbox
  │                                                  ↓
  │                                            ├─ SUCCESS ────────→  [analyze]  ──→  END
  │                                            │
  │                                            └─ ERROR ──────────→  [code_generation]  (retry)
  │                                                                   ↓ (if max_retries reached)
  │                                                                   [analyze]  ──→  END
  │
  └─ ERROR (missing data) ───→  [analyze]  ──→  Generate error response  ──→  END
```

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- E2B API key for code execution
- OpenAI or Anthropic API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/bio-xyz/bio-code-interpreter.git
cd bio-code-interpreter
```

2. Install dependencies using uv:

```bash
uv sync
```

3. Create a `.env` file with required API keys:

```bash
# Security - API key for authentication
API_KEY=your-secret-api-key-here

# Required environment for code execution
E2B_API_KEY=your_e2b_api_key

# OpenAI Configuration (required if using OpenAI provider)
OPENAI_API_KEY=your_openai_api_key

# Anthropic Configuration (required if using Anthropic provider)
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Running the Application

Start the FastAPI server with auto-reload:

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
app/
├── config/          # Application configuration and logging
├── models/          # Pydantic models and data structures
├── prompts/         # LLM prompt templates
├── routers/         # FastAPI route handlers
├── services/        # Core business logic
│   ├── agent/       # Agent FST implementation
│   │   ├── graph.py       # LangGraph workflow definition
│   │   ├── nodes.py       # FST state nodes
│   │   ├── state.py       # Agent state management
│   │   ├── transitions.py # State transition logic
│   │   └── signals.py     # Action signals
│   ├── llm/         # LLM service abstractions
│   └── executor_service.py # E2B code execution
└── utils/           # Utility functions

example/             # Example tasks and data files
```

## Usage Example

Submit a data analysis task with the dose-response example:

```bash
curl --location 'http://localhost:8000/api/task/run/sync' \
  --header 'X-API-Key: your-secret-api-key-here' \
  --form 'data_files=@"example/dose_response.csv"' \
  --form 'task_description="Given this CSV of drug concentrations and viability, fit a 4-parameter logistic curve, estimate IC50, and show the curve."'
```

## Configuration

Key settings can be configured via environment variables or in `app/config/settings.py`:

- `LLM_PROVIDER`: Choose between "openai" or "anthropic" (default: "anthropic")
- `LLM_MODEL`: Model name to use
- `CODE_GENERATION_MAX_RETRIES`: Maximum retry attempts for code generation (default: 3)
- `LOG_LEVEL`: Logging level (default: "INFO")

## Architecture

The agent follows a Finite State Transducer (FST) architecture with the following states:

1. **Plan**: Analyzes the task and creates an execution plan
2. **Code Generation**: Generates Python code based on the plan
3. **Execution**: Runs the code in an E2B sandbox
4. **Analyze**: Processes results and generates final response

State transitions are determined by action signals and execution feedback, enabling automatic error recovery and retry logic.
