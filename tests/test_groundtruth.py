from __future__ import annotations

import os
import pytest

from app.models.schemas import IntakeSubmission
from app.prompts.templates import HLD_PROMPT, INSIGHT_PROMPT, PRD_PROMPT, SPEC_PROMPT
from app.agents.workflow import GroundTruthEngine


# -----------------------------
# Fixtures / helpers
# -----------------------------

@pytest.fixture
def submission() -> IntakeSubmission:
    return IntakeSubmission(
        intake_type="service_request",
        title="Improve onboarding visibility",
        requested_by="qa-demo-user",
        business_context="Enterprise onboarding across multiple teams.",
        problem_statement="Teams lack visibility into milestones and blockers.",
        desired_outcome="Create a structured onboarding workflow with ownership and progress visibility.",
        raw_text=(
            "Title: Improve onboarding visibility\n"
            "Requested by: qa-demo-user\n"
            "Intake type: service_request\n"
            "Business context: Enterprise onboarding across multiple teams.\n"
            "Problem statement: Teams lack visibility into milestones and blockers.\n"
            "Desired outcome: Create a structured onboarding workflow with ownership and progress visibility."
        ),
    )


@pytest.fixture
def engine() -> GroundTruthEngine:
    return GroundTruthEngine()


# -----------------------------
# Unit tests
# -----------------------------

def test_01_intake_submission_accepts_supported_type(submission: IntakeSubmission):
    assert submission.intake_type == "service_request"
    assert submission.raw_text


def test_02_prompts_are_all_defined():
    assert "{submission}" in INSIGHT_PROMPT
    assert "{submission}" in SPEC_PROMPT
    assert "{spec}" in HLD_PROMPT
    assert "{hld}" in PRD_PROMPT


def test_03_langfuse_invoke_config_includes_metadata(engine: GroundTruthEngine, submission: IntakeSubmission):
    cfg = engine._build_invoke_config(
        submission=submission,
        session_id="gt-session-001",
        user_id="qa-user-001",
        tags=["groundtruth", "pytest"],
    )
    assert cfg["metadata"]["langfuse_session_id"] == "gt-session-001"
    assert cfg["metadata"]["langfuse_user_id"] == "qa-user-001"
    assert cfg["metadata"]["intake_type"] == "service_request"
    assert cfg["metadata"]["langfuse_tags"] == ["groundtruth", "pytest"]


def test_04_workflow_falls_back_when_optional_fields_missing(engine: GroundTruthEngine):
    submission = IntakeSubmission(
        intake_type="customer_bug",
        raw_text="Customer reports save failure after refresh.",
    )
    cfg = engine._build_invoke_config(submission=submission)
    assert cfg["metadata"]["intake_type"] == "customer_bug"
    assert cfg["metadata"]["submission_title"] == ""
    assert cfg["metadata"]["langfuse_user_id"] == "Anonymous User"


def test_05_langfuse_callbacks_disabled_without_keys(monkeypatch, engine: GroundTruthEngine):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert engine._langfuse_callbacks() == []


def test_06_submission_to_text_contains_problem_statement(engine: GroundTruthEngine, submission: IntakeSubmission):
    payload = engine._submission_to_text(submission)
    assert "problem_statement" in payload
    assert "milestones and blockers" in payload


def test_07_markdown_export_contains_expected_sections(submission: IntakeSubmission):
    from app.models.schemas import (
        ExecutableSpecification,
        GroundTruthOutput,
        HighLevelDesign,
        InsightSummary,
        PRD,
        PRDSection,
    )

    output = GroundTruthOutput(
        submission=submission,
        insight=InsightSummary(summary="Insight", key_themes=["Theme A"], risks=["Risk A"], recommendations=["Rec A"]),
        executable_spec=ExecutableSpecification(
            summary="Spec",
            functional_requirements=["FR1"],
            non_functional_requirements=["NFR1"],
            acceptance_criteria=["AC1"],
        ),
        high_level_design=HighLevelDesign(
            overview="HLD",
            components=["UI", "Workflow"],
            data_flow=["UI -> Workflow"],
            dependencies=["Gemini"],
        ),
        prd=PRD(summary="PRD", sections=[PRDSection(title="Goals", content="Improve visibility")]),
    )

    md = GroundTruthEngine().to_markdown(output)
    assert "## Insight Summary" in md
    assert "## Executable Specification" in md
    assert "## High-Level Design" in md
    assert "## Product Requirements Document" in md


def test_08_supported_intake_types_are_limited():
    valid = {"service_request", "customer_bug", "feature_request", "competitor_insight"}
    assert len(valid) == 4


# -----------------------------
# Optional integration tests
# -----------------------------

@pytest.mark.skipif(not os.getenv("LANGSMITH_API_KEY"), reason="LANGSMITH_API_KEY not configured")
def test_09_langsmith_client_connectivity():
    from langsmith import Client

    client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    assert client is not None
    assert hasattr(client, "list_runs") or hasattr(client, "create_run")


@pytest.mark.skipif(not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")), reason="Langfuse keys not configured")
def test_10_langfuse_api_connectivity():
    from langfuse import get_client

    client = get_client()
    assert client is not None
    assert hasattr(client, "api")
    assert hasattr(client.api, "observations") or hasattr(client.api, "metrics")
