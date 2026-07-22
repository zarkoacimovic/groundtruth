from __future__ import annotations

import os
import uuid
import pytest

from app.models.schemas import IntakeSubmission
from app.agents.workflow import GroundTruthEngine


# -----------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------

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


# -----------------------------------------------------------------
# Skip rules
# -----------------------------------------------------------------

live_eval_required = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="Live Gemini evaluation requires GOOGLE_API_KEY or GEMINI_API_KEY",
)


def _langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


# -----------------------------------------------------------------
# Helper: record a numeric quality score in Langfuse
# -----------------------------------------------------------------

def record_langfuse_score(
    session_id: str,
    name: str,
    value: float,
    comment: str = "",
) -> None:
    """Send a quality score to Langfuse for the given session.

    Uses the Langfuse Python SDK. Silently no-ops if Langfuse is not configured
    so tests still run in environments without Langfuse credentials.
    """
    if not _langfuse_enabled():
        return

    try:
        from langfuse import get_client

        client = get_client()

        client.create_score(
            name=name,
            value=value,
            comment=comment or None,
            session_id=session_id,
            data_type="NUMERIC",
        )

        # Ensure scores are actually flushed before the test process exits.
        try:
            client.flush()
        except Exception:
            pass
    except Exception as exc:
        # Never let scoring break the test itself.
        print(f"[langfuse-score-warning] Could not record score '{name}': {exc}")


# -----------------------------------------------------------------
# Quality-check helpers
# -----------------------------------------------------------------

def _relevance_signals(text: str, allowed_keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for keyword in allowed_keywords if keyword in text_lower)


def _score_boolean(passed: bool) -> float:
    return 1.0 if passed else 0.0


# -----------------------------------------------------------------
# Live evaluation tests with Langfuse scoring
# -----------------------------------------------------------------

@live_eval_required
def test_08_live_eval_insight_is_non_empty_and_relevant(submission: IntakeSubmission):
    session_id = f"live-eval-insight-{uuid.uuid4().hex[:8]}"

    engine = GroundTruthEngine()
    result = engine.run(submission, session_id=session_id, user_id="qa-live")

    insight = result.insight
    allowed_keywords = [
        "onboarding",
        "visibility",
        "milestones",
        "blockers",
        "ownership",
        "progress",
    ]

    non_empty = insight.summary.strip() != ""
    has_theme = len(insight.key_themes) >= 1
    relevance_count = _relevance_signals(insight.summary, allowed_keywords)
    relevance_ok = relevance_count >= 1

    # Record Langfuse scores
    record_langfuse_score(session_id, "insight_non_empty", _score_boolean(non_empty))
    record_langfuse_score(session_id, "insight_has_theme", _score_boolean(has_theme))
    record_langfuse_score(session_id, "insight_relevance_score", float(relevance_count),
                          comment=f"keywords_hit={relevance_count}/{len(allowed_keywords)}")
    record_langfuse_score(session_id, "insight_overall_pass",
                          _score_boolean(non_empty and has_theme and relevance_ok))

    assert non_empty, "Insight summary was empty"
    assert has_theme, "Insight has no key themes"
    assert relevance_ok, f"Insight did not include any expected keywords: {insight.summary}"


@live_eval_required
def test_09_live_eval_spec_contains_testable_requirements(submission: IntakeSubmission):
    session_id = f"live-eval-spec-{uuid.uuid4().hex[:8]}"

    engine = GroundTruthEngine()
    result = engine.run(submission, session_id=session_id, user_id="qa-live")

    spec = result.executable_spec

    non_empty_summary = spec.summary.strip() != ""
    enough_functional = len(spec.functional_requirements) >= 2
    has_acceptance_criteria = len(spec.acceptance_criteria) >= 1

    combined_ac = " ".join(spec.acceptance_criteria).lower()
    quality_signals = ["shall", "must", "when", "then", "display", "track", "assign"]
    quality_hits = sum(1 for signal in quality_signals if signal in combined_ac)
    testable_language_ok = quality_hits >= 1

    record_langfuse_score(session_id, "spec_summary_non_empty", _score_boolean(non_empty_summary))
    record_langfuse_score(session_id, "spec_functional_requirements_count",
                          float(len(spec.functional_requirements)))
    record_langfuse_score(session_id, "spec_acceptance_criteria_count",
                          float(len(spec.acceptance_criteria)))
    record_langfuse_score(session_id, "spec_testable_language_hits", float(quality_hits))
    record_langfuse_score(session_id, "spec_overall_pass",
                          _score_boolean(
                              non_empty_summary
                              and enough_functional
                              and has_acceptance_criteria
                              and testable_language_ok
                          ))

    assert non_empty_summary, "Spec summary was empty"
    assert enough_functional, "Spec had fewer than 2 functional requirements"
    assert has_acceptance_criteria, "Spec had no acceptance criteria"
    assert testable_language_ok, (
        "Acceptance criteria did not contain any testable/actionable language "
        f"signals: {spec.acceptance_criteria}"
    )


@live_eval_required
def test_10_live_eval_bug_request_produces_actionable_outputs(bug_submission: IntakeSubmission):
    session_id = f"live-eval-bug-{uuid.uuid4().hex[:8]}"

    engine = GroundTruthEngine()
    result = engine.run(bug_submission, session_id=session_id, user_id="qa-live")

    insight_non_empty = result.insight.summary.strip() != ""
    hld_non_empty = result.high_level_design.overview.strip() != ""
    prd_has_sections = len(result.prd.sections) >= 1

    joined_content = " ".join(
        [result.insight.summary, result.high_level_design.overview, result.prd.summary]
        + [section.title + " " + section.content for section in result.prd.sections]
    ).lower()

    bug_signals = ["save", "refresh", "data", "persistence", "state", "update", "lost"]
    bug_hits = sum(1 for signal in bug_signals if signal in joined_content)
    bug_context_ok = bug_hits >= 1

    record_langfuse_score(session_id, "bug_insight_non_empty", _score_boolean(insight_non_empty))
    record_langfuse_score(session_id, "bug_hld_non_empty", _score_boolean(hld_non_empty))
    record_langfuse_score(session_id, "bug_prd_section_count",
                          float(len(result.prd.sections)))
    record_langfuse_score(session_id, "bug_context_signal_count", float(bug_hits))
    record_langfuse_score(session_id, "bug_overall_pass",
                          _score_boolean(
                              insight_non_empty
                              and hld_non_empty
                              and prd_has_sections
                              and bug_context_ok
                          ))

    assert insight_non_empty, "Bug insight was empty"
    assert hld_non_empty, "Bug HLD overview was empty"
    assert prd_has_sections, "Bug PRD had no sections"
    assert bug_context_ok, (
        "Bug output did not include bug-context signals such as 'save', 'refresh', "
        f"'persistence', 'state', 'data': {joined_content[:400]}"
    )
