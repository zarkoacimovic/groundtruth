from __future__ import annotations

import json
from typing import Any, Dict, TypedDict

from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from app.models.schemas import (
    ExecutableSpecification,
    GroundTruthOutput,
    HighLevelDesign,
    InsightSummary,
    IntakeSubmission,
    PRD,
)
from app.prompts.templates import SYSTEM_PROMPT
from app.utils.config import get_model_name

try:
    from langfuse.callback import CallbackHandler
except Exception:
    CallbackHandler = None


FULL_ARTIFACT_PROMPT = """
You are GroundTruth, an AI product operations copilot.

Transform the PM intake into four grounded artifacts in ONE response:
1. insight
2. executable_spec
3. high_level_design
4. prd

Rules:
- Be concise, practical, and implementation-oriented.
- Do not invent unsupported facts.
- If details are missing, include them under assumptions, risks, or open_questions.
- Keep outputs useful for product, engineering, and QA collaboration.

Input type: {intake_type}

Raw PM text:
{raw_text}

Return structured output with these sections:

insight:
- title
- problem_statement
- business_impact
- user_segments
- assumptions
- risks

executable_spec:
- summary
- user_stories
- acceptance_criteria
- test_scenarios
- non_functional_requirements

high_level_design:
- architecture_overview
- components
- interfaces
- data_flow
- observability

prd:
- objective
- success_metrics
- scope_in
- scope_out
- rollout_notes
- open_questions
""".strip()


class SingleCallArtifacts(BaseModel):
    insight: InsightSummary
    executable_spec: ExecutableSpecification
    high_level_design: HighLevelDesign
    prd: PRD


class GroundTruthState(TypedDict, total=False):
    submission: IntakeSubmission
    artifacts: SingleCallArtifacts


def _langfuse_callbacks() -> list:
    if CallbackHandler is None:
        return []
    try:
        return [CallbackHandler()]
    except Exception:
        return []


class GroundTruthEngine:
    def __init__(self) -> None:
        self.model = ChatGoogleGenerativeAI(
            model=get_model_name(),
            temperature=0.2,
        )
        self.callbacks = _langfuse_callbacks()
        self.graph = self._build_graph()

    def _invoke_structured(self, schema: Any, prompt: str) -> Any:
        chain = self.model.with_structured_output(schema)
        return chain.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)],
            config={
                "callbacks": self.callbacks,
                "metadata": {"app": "groundtruth-mvp"},
                "tags": ["groundtruth", "single_call", schema.__name__.lower()],
            },
        )

    def _build_graph(self):
        graph = StateGraph(GroundTruthState)
        graph.add_node("generate_all", self._generate_all)

        graph.add_edge(START, "generate_all")
        graph.add_edge("generate_all", END)
        return graph.compile()

    def _generate_all(self, state: GroundTruthState) -> Dict[str, Any]:
        submission = state["submission"]
        prompt = FULL_ARTIFACT_PROMPT.format(
            intake_type=submission.intake_type.value,
            raw_text=submission.raw_text,
        )
        artifacts = self._invoke_structured(SingleCallArtifacts, prompt)
        return {"artifacts": artifacts}

    def run(self, submission: IntakeSubmission) -> GroundTruthOutput:
        result = self.graph.invoke(
            {"submission": submission},
            config={"callbacks": self.callbacks, "run_name": "groundtruth_pipeline"},
        )
        artifacts = result["artifacts"]

        return GroundTruthOutput(
            intake=submission.model_dump(),
            insight=artifacts.insight.model_dump(),
            executable_spec=artifacts.executable_spec.model_dump(),
            high_level_design=artifacts.high_level_design.model_dump(),
            prd=artifacts.prd.model_dump(),
        )

    @staticmethod
    def to_markdown(output: GroundTruthOutput) -> str:
        blocks = [
            f"# GroundTruth Output: {output.insight.title}",
            "## Intake",
            f"- Type: {output.intake.intake_type.value}",
            f"- Submitted at: {output.intake.submitted_at}",
            "",
            "## Insight Summary",
            json.dumps(output.insight.model_dump(), indent=2),
            "",
            "## Executable Specification",
            json.dumps(output.executable_spec.model_dump(), indent=2),
            "",
            "## High-Level Design",
            json.dumps(output.high_level_design.model_dump(), indent=2),
            "",
            "## PRD",
            json.dumps(output.prd.model_dump(), indent=2),
        ]
        return "\n".join(blocks)
