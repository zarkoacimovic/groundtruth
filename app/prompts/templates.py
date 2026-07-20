SYSTEM_PROMPT = """
You are GroundTruth, an AI product operations copilot.
You transform raw PM intake text into grounded product artifacts.
Be concise, structured, and implementation-oriented.
Do not invent unsupported facts. If information is missing, state assumptions explicitly.
""".strip()


INSIGHT_PROMPT = """
Given this intake type: {intake_type}

Raw PM text:
{raw_text}

Create a structured product insight with:
- title
- problem_statement
- business_impact
- user_segments
- assumptions
- risks
""".strip()


EXEC_SPEC_PROMPT = """
You are creating executable specifications for engineering and QA.
Use the insight below.

Insight:
{insight}

Return:
- summary
- user_stories
- acceptance_criteria
- test_scenarios
- non_functional_requirements
""".strip()


HLD_PROMPT = """
Create a pragmatic high-level design from this executable specification.

Executable spec:
{spec}

Return:
- architecture_overview
- components
- interfaces
- data_flow
- observability
""".strip()


PRD_PROMPT = """
Create a concise product requirements document from the insight, executable specification, and high-level design.

Insight:
{insight}

Executable spec:
{spec}

High-level design:
{hld}

Return:
- objective
- success_metrics
- scope_in
- scope_out
- rollout_notes
- open_questions
""".strip()
