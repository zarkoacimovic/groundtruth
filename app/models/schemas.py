from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


IntakeType = Literal[
    "service_request",
    "customer_bug",
    "feature_request",
    "competitor_insight",
]


class IntakeSubmission(BaseModel):
    """
    Normalized intake payload used by the GroundTruth MVP.

    This schema is intentionally forgiving and aligned with app/main.py and
    app/agents/workflow.py. It supports the four MVP intake types and includes
    both structured fields and a raw_text field for prompt grounding,
    observability, and debugging.
    """

    intake_type: IntakeType
    raw_text: str

    title: str = ""
    requested_by: str = "Anonymous User"
    business_context: str = ""
    problem_statement: str = ""
    desired_outcome: str = ""


class InsightSummary(BaseModel):
    summary: str = ""
    key_themes: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class ExecutableSpecification(BaseModel):
    summary: str = ""
    functional_requirements: List[str] = Field(default_factory=list)
    non_functional_requirements: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)


class HighLevelDesign(BaseModel):
    overview: str = ""
    components: List[str] = Field(default_factory=list)
    data_flow: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)


class PRDSection(BaseModel):
    title: str = ""
    content: str = ""


class PRD(BaseModel):
    summary: str = ""
    sections: List[PRDSection] = Field(default_factory=list)


class GroundTruthOutput(BaseModel):
    submission: IntakeSubmission
    insight: InsightSummary
    executable_spec: ExecutableSpecification
    high_level_design: HighLevelDesign
    prd: PRD
