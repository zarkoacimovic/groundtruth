from __future__ import annotations

import os
import pytest

from app.models.schemas import IntakeSubmission
from app.prompts.templates import HLD_PROMPT, INSIGHT_PROMPT, PRD_PROMPT, SPEC_PROMPT
from app.agents.workflow import GroundTruthEngine


# ---------------------------------------------------------------------
# Test data / fixtures
# ---------------------------------------------------------------------

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
def bug_submission() -> IntakeSubmission:
    return IntakeSubmission(
        intake_type="customer_bug",
        title="Save action fails after refresh",
        requested_by="support-team",
        business_context="A customer support team is handling repeated complaints from enterprise users.",
        problem_statement="Users report that edits appear to save, but after a page refresh the changes are lost.",
        desired_outcome="Identify the likely defect scope and generate a concrete implementation-ready fix plan.",
        raw_text=(
            "Title: Save action fails after refresh\n"
            "Requested by: support-team\n"
            "Intake type: customer_bug\n"
            "Business context: A customer support team is handling repeated complaints from enterprise users.\n"
            "Problem statement: Users report that edits appear to save, but after a page refresh the changes are lost.\n"
            "Desired outcome: Identify the likely defect scope and generate a concrete implementation-ready fix plan."
        ),
    )


@pytest.fixture
def engine(monkeypatch) -> GroundTruthEngine:
    # Fast/local tests should not require a real Gemini key.
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-api-key")
    monkeypatch.setenv("GROUNDTRUTH_MODEL", "gemini-3.5-flash")
    return GroundTruthEngine()


# ---------------------------------------------------------------------
# Fast structural / wiring tests
# ---------------------------------------------------------------------

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
    minimal = IntakeSubmission(
        intake_type="customer_bug",
        raw_text="Customer reports save failure after refresh.",
    )
    cfg = engine._build_invoke_config(submission=minimal)
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


def test_07_markdown_export_contains_expected_sections(monkeypatch, submission: IntakeSubmission):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-api-key")

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


# ---------------------------------------------------------------------
# Live evaluation tests (real Gemini calls)
# These tests require GOOGLE_API_KEY and will consume API quota.
# ---------------------------------------------------------------------

live_eval_required = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="Live Gemini evaluation requires GOOGLE_API_KEY or GEMINI_API_KEY",
)


@live_eval_required
def test_08_live_eval_insight_is_non_empty_and_relevant(submission: IntakeSubmission):
    engine = GroundTruthEngine()
    result = engine.run(submission, session_id="live-eval-insight", user_id="qa-live")

    assert result.insight.summary.strip() != ""
    assert len(result.insight.key_themes) >= 1

    summary_lower = result.insight.summary.lower()
    allowed_keywords = [
        "onboarding",
        "visibility",
        "milestones",
        "blockers",
        "ownership",
        "progress",
    ]
    assert any(keyword in summary_lower for keyword in allowed_keywords), result.insight.summary


@live_eval_required
def test_09_live_eval_spec_contains_testable_requirements(submission: IntakeSubmission):
    engine = GroundTruthEngine()
    result = engine.run(submission, session_id="live-eval-spec", user_id="qa-live")

    assert result.executable_spec.summary.strip() != ""
    assert len(result.executable_spec.functional_requirements) >= 2
    assert len(result.executable_spec.acceptance_criteria) >= 1

    combined = " ".join(result.executable_spec.acceptance_criteria).lower()
    quality_signals = ["shall", "must", "when", "then", "display", "track", "assign"]
    assert any(signal in combined for signal in quality_signals), result.executable_spec.acceptance_criteria


@live_eval_required
def test_10_live_eval_bug_request_produces_actionable_outputs(bug_submission: IntakeSubmission):
    engine = GroundTruthEngine()
    result = engine.run(bug_submission, session_id="live-eval-bug", user_id="qa-live")

    assert result.insight.summary.strip() != ""
    assert result.high_level_design.overview.strip() != ""
    assert len(result.prd.sections) >= 1

    joined_content = " ".join(
        [result.insight.summary, result.high_level_design.overview, result.prd.summary]
        + [section.title + " " + section.content for section in result.prd.sections]
    ).lower()

    bug_signals = ["save", "refresh", "data", "persistence", "state", "update", "lost"]
    assert any(signal in joined_content for signal in bug_signals), joined_content
