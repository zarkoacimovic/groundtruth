from __future__ import annotations

import os
import uuid
from typing import Literal

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.agents.workflow import GroundTruthEngine
from app.models.schemas import GroundTruthOutput, IntakeSubmission


# -----------------------------------------------------------------
# Skip rules
# -----------------------------------------------------------------

semantic_eval_required = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="Semantic Gemini evaluation requires GOOGLE_API_KEY or GEMINI_API_KEY",
)


# -----------------------------------------------------------------
# Semantic judge schema
# -----------------------------------------------------------------

class SemanticJudgeResult(BaseModel):
    intent_match: float = Field(ge=0.0, le=1.0)
    context_preservation: float = Field(ge=0.0, le=1.0)
    actionability: float = Field(ge=0.0, le=1.0)
    cross_artifact_consistency: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    verdict: Literal["pass", "warning", "fail"]
    rationale: str = ""
    must_fix: list[str] = Field(default_factory=list)


# -----------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------

@pytest.fixture
def service_submission() -> IntakeSubmission:
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
def feature_submission() -> IntakeSubmission:
    return IntakeSubmission(
        intake_type="feature_request",
        title="Add release note approval workflow",
        requested_by="product-ops",
        business_context="Multiple teams publish release notes, but there is no consistent review or sign-off process.",
        problem_statement="Teams ship release notes without approval, creating inconsistent messaging and audit gaps.",
        desired_outcome="Design a clear approval workflow with ownership, status tracking, and Slack notifications.",
        raw_text=(
            "Title: Add release note approval workflow\n"
            "Requested by: product-ops\n"
            "Intake type: feature_request\n"
            "Business context: Multiple teams publish release notes, but there is no consistent review or sign-off process.\n"
            "Problem statement: Teams ship release notes without approval, creating inconsistent messaging and audit gaps.\n"
            "Desired outcome: Design a clear approval workflow with ownership, status tracking, and Slack notifications."
        ),
    )


# -----------------------------------------------------------------
# Langfuse helpers
# -----------------------------------------------------------------

def _langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def record_langfuse_score(session_id: str, name: str, value: float, comment: str = "") -> None:
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
        try:
            client.flush()
        except Exception:
            pass
    except Exception as exc:
        print(f"[langfuse-score-warning] Could not record score '{name}': {exc}")


# -----------------------------------------------------------------
# Semantic judge helper
# -----------------------------------------------------------------

def _judge_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GROUNDTRUTH_JUDGE_MODEL", os.getenv("GROUNDTRUTH_MODEL", "gemini-3.5-flash")),
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    )


def judge_groundtruth_output(
    *,
    rubric_name: str,
    rubric_text: str,
    submission: IntakeSubmission,
    output: GroundTruthOutput,
) -> SemanticJudgeResult:
    prompt = ChatPromptTemplate.from_template(
        """
You are an expert QA evaluator for a product-intake transformation system.

Your job is to evaluate whether the generated output is semantically correct for the original intake.

Rubric name:
{rubric_name}

Rubric:
{rubric_text}

Scoring instructions:
- Score each numeric field from 0.0 to 1.0.
- Use 1.0 only if the dimension is clearly strong.
- Use lower scores if the output is generic, drifts from the request, invents unsupported assumptions, or contradicts itself.
- groundedness means the output stays anchored to facts/details present in the submission.
- actionability means the output could realistically help a PM, engineer, or QA person act.
- cross_artifact_consistency means insight, spec, HLD, and PRD tell the same story.
- verdict must be one of: pass, warning, fail.
- must_fix should contain short bullet-style issues only when there are meaningful problems.

Original submission:
{submission_json}

Generated output:
{output_json}
"""
    )

    chain = prompt | _judge_model().with_structured_output(SemanticJudgeResult)
    return chain.invoke(
        {
            "rubric_name": rubric_name,
            "rubric_text": rubric_text,
            "submission_json": submission.model_dump_json(indent=2),
            "output_json": output.model_dump_json(indent=2),
        }
    )


def run_semantic_eval(
    *,
    session_id: str,
    rubric_name: str,
    rubric_text: str,
    submission: IntakeSubmission,
) -> tuple[GroundTruthOutput, SemanticJudgeResult]:
    engine = GroundTruthEngine()
    output = engine.run(submission, session_id=session_id, user_id="qa-semantic")
    judge = judge_groundtruth_output(
        rubric_name=rubric_name,
        rubric_text=rubric_text,
        submission=submission,
        output=output,
    )

    record_langfuse_score(session_id, f"{rubric_name}_intent_match", judge.intent_match)
    record_langfuse_score(session_id, f"{rubric_name}_context_preservation", judge.context_preservation)
    record_langfuse_score(session_id, f"{rubric_name}_actionability", judge.actionability)
    record_langfuse_score(session_id, f"{rubric_name}_cross_artifact_consistency", judge.cross_artifact_consistency)
    record_langfuse_score(session_id, f"{rubric_name}_groundedness", judge.groundedness)
    record_langfuse_score(
        session_id,
        f"{rubric_name}_overall_score",
        judge.overall_score,
        comment=f"verdict={judge.verdict}; rationale={judge.rationale[:250]}",
    )

    return output, judge


# -----------------------------------------------------------------
# Semantic evaluation tests
# -----------------------------------------------------------------

@semantic_eval_required
@pytest.mark.semantic_eval
def test_11_semantic_service_request_alignment(service_submission: IntakeSubmission):
    session_id = f"semantic-service-{uuid.uuid4().hex[:8]}"
    rubric_name = "service_request_semantics"
    rubric_text = (
        "Evaluate whether the output correctly interprets this as a service/operational workflow request. "
        "The output should stay focused on onboarding visibility, ownership, milestones, blockers, and progress tracking. "
        "Penalize generic product strategy, invented technical systems not implied by the intake, or outputs that ignore workflow visibility."
    )

    _, judge = run_semantic_eval(
        session_id=session_id,
        rubric_name=rubric_name,
        rubric_text=rubric_text,
        submission=service_submission,
    )

    assert judge.intent_match >= 0.80, judge.model_dump_json(indent=2)
    assert judge.context_preservation >= 0.75, judge.model_dump_json(indent=2)
    assert judge.actionability >= 0.70, judge.model_dump_json(indent=2)
    assert judge.groundedness >= 0.70, judge.model_dump_json(indent=2)
    assert judge.overall_score >= 0.75, judge.model_dump_json(indent=2)
    assert judge.verdict in {"pass", "warning"}, judge.model_dump_json(indent=2)


@semantic_eval_required
@pytest.mark.semantic_eval
def test_12_semantic_bug_context_preservation(bug_submission: IntakeSubmission):
    session_id = f"semantic-bug-{uuid.uuid4().hex[:8]}"
    rubric_name = "bug_context_semantics"
    rubric_text = (
        "Evaluate whether the output correctly stays in bug-analysis mode for a persistence/state-loss defect. "
        "A strong output should preserve the save/refresh failure mode, identify likely defect scope, and propose a fix-oriented plan. "
        "Penalize reframing the request as a new feature, ignoring the persistence problem, or inventing unsupported root causes."
    )

    _, judge = run_semantic_eval(
        session_id=session_id,
        rubric_name=rubric_name,
        rubric_text=rubric_text,
        submission=bug_submission,
    )

    assert judge.intent_match >= 0.85, judge.model_dump_json(indent=2)
    assert judge.context_preservation >= 0.80, judge.model_dump_json(indent=2)
    assert judge.actionability >= 0.75, judge.model_dump_json(indent=2)
    assert judge.groundedness >= 0.75, judge.model_dump_json(indent=2)
    assert judge.overall_score >= 0.78, judge.model_dump_json(indent=2)
    assert judge.verdict in {"pass", "warning"}, judge.model_dump_json(indent=2)


@semantic_eval_required
@pytest.mark.semantic_eval
def test_13_semantic_feature_request_buildability(feature_submission: IntakeSubmission):
    session_id = f"semantic-feature-{uuid.uuid4().hex[:8]}"
    rubric_name = "feature_request_semantics"
    rubric_text = (
        "Evaluate whether the output correctly interprets this as a feature request for an approval workflow. "
        "A strong output should preserve the need for approval, ownership, status tracking, auditability, and Slack notifications. "
        "The generated spec and HLD should be buildable and internally consistent. Penalize vague product prose, bug-style reframing, or unsupported assumptions."
    )

    _, judge = run_semantic_eval(
        session_id=session_id,
        rubric_name=rubric_name,
        rubric_text=rubric_text,
        submission=feature_submission,
    )

    assert judge.intent_match >= 0.80, judge.model_dump_json(indent=2)
    assert judge.actionability >= 0.75, judge.model_dump_json(indent=2)
    assert judge.cross_artifact_consistency >= 0.75, judge.model_dump_json(indent=2)
    assert judge.groundedness >= 0.70, judge.model_dump_json(indent=2)
    assert judge.overall_score >= 0.75, judge.model_dump_json(indent=2)
    assert judge.verdict in {"pass", "warning"}, judge.model_dump_json(indent=2)
