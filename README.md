# GroundTruth MVP

GroundTruth is a simple PM-intake-to-artifacts MVP built with **LangChain**, **LangGraph**, **LangSmith**, and **Langfuse**.

## What it does

GroundTruth provides four very simple Voice of Customer / product intake forms:

1. Service Requests
2. Customer Found Bugs
3. New Feature Requests
4. Competitor Insights

A project manager can paste raw text into any form. GroundTruth then generates:

- **Executable Specifications**
- **High-Level Design (HLD)**
- **Product Requirements Document (PRD)**

The current version uses simple forms only, while keeping clear extension points for future integrations with:

- Slack
- ServiceNow
- Jira

## Architecture

- **LangChain**: model abstraction and structured output
- **LangGraph**: orchestration pipeline
- **LangSmith**: tracing/evaluation via environment variables
- **Langfuse**: callbacks/observability
- **Gemini**: LLM provider through `langchain-google-genai`
- **Streamlit**: MVP UI

## Workflow

```text
PM text input
  -> Insight extraction
  -> Executable specification generation
  -> High-level design generation
  -> PRD generation
```

## Project structure

```text
app/
  agents/workflow.py          # LangGraph pipeline
  models/schemas.py           # Pydantic schemas
  prompts/templates.py        # Prompt templates
  integrations/placeholders.py# Future integration stubs
  utils/config.py             # Env/config helpers
  main.py                     # Streamlit app
```

## Setup

1. Create and activate a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy env template:

```bash
cp .env.example .env
```

4. Fill in your secrets locally (do not commit them)

## Required environment variables

```env
GOOGLE_API_KEY=your_gemini_api_key

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=groundtruth-dev

LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

GROUNDTRUTH_MODEL=gemini-1.5-flash
```

## Run

```bash
streamlit run app/main.py
```

## Future roadmap

- Slack ingestion adapter
- ServiceNow intake adapter
- Jira sync for generated tickets and specs
- LangSmith eval datasets for regression testing
- Langfuse dashboards for cost/latency/product telemetry
- Authentication and artifact history

## Notes

- This MVP intentionally keeps the forms extremely simple.
- The generated outputs are grounded only in the text provided by the PM.
- If details are missing, the system states assumptions explicitly.
