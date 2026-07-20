from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class IntakeType(str, Enum):
    SERVICE_REQUEST = "service_request"
    CUSTOMER_BUG = "customer_bug"
    FEATURE_REQUEST = "feature_request"
    COMPETITOR_INSIGHT = "competitor_insight"


class IntakeSubmission(BaseModel):
    intake_type: IntakeType
    raw_text: str = Field(..., min_length=10)
    submitted_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class InsightSummary(BaseModel):
    title: str
    problem_statement: str
    business_impact: str
    user_segments: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)


class ExecutableSpecification(BaseModel):
    summary: str
    user_stories: List[str]
    acceptance_criteria: List[str]
    test_scenarios: List[str]
    non_functional_requirements: List[str]


class HighLevelDesign(BaseModel):
    architecture_overview: str
    components: List[str]
    interfaces: List[str]
    data_flow: List[str]
    observability: List[str]


class PRD(BaseModel):
    objective: str
    success_metrics: List[str]
    scope_in: List[str]
    scope_out: List[str]
    rollout_notes: List[str]
    open_questions: List[str]


class GroundTruthOutput(BaseModel):
    intake: IntakeSubmission
    insight: InsightSummary
    executable_spec: ExecutableSpecification
    high_level_design: HighLevelDesign
    prd: PRD
