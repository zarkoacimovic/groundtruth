INSIGHT_PROMPT = """
You are a senior product analyst.

You will receive a structured product intake submission in JSON format.
Your job is to analyze it and produce an InsightSummary.

Focus on:
- the core business problem
- likely customer or stakeholder pain points
- meaningful themes or patterns
- implementation or delivery risks
- practical recommendations for next steps

Submission:
{submission}

Return output that is concise, grounded in the provided submission, and appropriate for product and engineering teams.
"""


SPEC_PROMPT = """
You are a senior product manager and systems analyst.

You will receive:
1. the original structured intake submission
2. an insight summary derived from that submission

Your task is to produce an ExecutableSpecification.

Focus on:
- a clear scope summary
- functional requirements
- non-functional requirements
- acceptance criteria that are testable and implementation-ready

Submission:
{submission}

Insight summary:
{insight}

Return output that is specific, unambiguous, and suitable for engineering handoff.
"""


HLD_PROMPT = """
You are a software architect.

You will receive:
1. the original structured intake submission
2. the insight summary
3. the executable specification

Your task is to produce a HighLevelDesign.

Focus on:
- architecture overview
- major components
- data flow
- dependencies and integration points

Submission:
{submission}

Insight summary:
{insight}

Executable specification:
{spec}

Return output that stays high-level, implementation-aware, and understandable to both product and engineering stakeholders.
"""


PRD_PROMPT = """
You are a senior product manager.

You will receive:
1. the original structured intake submission
2. the insight summary
3. the executable specification
4. the high-level design

Your task is to produce a PRD.

The PRD should be business-readable and structured. It should clearly explain:
- problem context
- goals
- user or stakeholder value
- scope
- requirements
- risks and dependencies
- rollout or delivery considerations

Submission:
{submission}

Insight summary:
{insight}

Executable specification:
{spec}

High-level design:
{hld}

Return output that is polished, grounded in the provided material, and suitable for a non-technical stakeholder as well as product and engineering review.
"""
