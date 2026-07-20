from __future__ import annotations

import json
import os
from typing import Any, Dict, TypedDict

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
from app.prompts.templates import (
    EXEC_SPEC_PROMPT,
    HLD_PROMPT,
    INSIGHT_PROMPT,
    PRD_PROMPT,
    SYSTEM_PROMPT,
)
from app.utils.config import get_model_name

try:
    from langfuse import Langfuse, get_client
    from langfuse.langchain import CallbackHandler
except Exception:  # pragma: no cover
    Langfuse = None
    get_client = None
    CallbackHandler = None


class GroundTruthState(TypedDict, total=False):
    submission: IntakeSubmission
    insight: InsightSummary
    executable_spec: ExecutableSpecification
    high_level_design: HighLevelDesign
    prd: PRD


def _langfuse_callbacks() -> list:
    if CallbackHandler is None or Langfuse is None:
        return []

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    if not public_key or not secret_key:
        return []

    try:
        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        get_client()
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
                "metadata": {
                    "app": "groundtruth-mvp",
                    "langfuse_session_id": "groundtruth-session",
                },
                "tags": ["groundtruth", schema.__name__.lower()],
            },
        )

    def _build_graph(self):
        graph = StateGraph(GroundTruthState)
        graph.add_node("insight", self._create_insight)
        graph.add_node("spec", self._create_spec)
        graph.add_node("hld", self._create_hld)
        graph.add_node("prd", self._create_prd)

        graph.add_edge(START, "insight")
        graph.add_edge("insight", "spec")
        graph.add_edge("spec", "hld")
        graph.add_edge("hld", "prd")
        graph.add_edge("prd", END)
        return graph.compile()

    def _create_insight(self, state: GroundTruthState) -> Dict[str, Any]:
        submission = state["submission"]
        prompt = INSIGHT_PROMPT.format(
            intake_type=submission.intake_type.value,
            raw_text=submission.raw_text,
        )
        insight = self._invoke_structured(InsightSummary, prompt)
        return {"insight": insight}

    def _create_spec(self, state: GroundTruthState) -> Dict[str, Any]:
        prompt = EXEC_SPEC_PROMPT.format(
            insight=state["insight"].model_dump_json(indent=2),
        )
        spec = self._invoke_structured(ExecutableSpecification, prompt)
        return {"executable_spec": spec}

    def _create_hld(self, state: GroundTruthState) -> Dict[str, Any]:
        prompt = HLD_PROMPT.format(
            spec=state["executable_spec"].model_dump_json(indent=2),
        )
        hld = self._invoke_structured(HighLevelDesign, prompt)
        return {"high_level_design": hld}

    def _create_prd(self, state: GroundTruthState) -> Dict[str, Any]:
        prompt = PRD_PROMPT.format(
            insight=state["insight"].model_dump_json(indent=2),
            spec=state["executable_spec"].model_dump_json(indent=2),
            hld=state["high_level_design"].model_dump_json(indent=2),
        )
        prd = self._invoke_structured(PRD, prompt)
        return {"prd": prd}

    def run(self, submission: IntakeSubmission) -> GroundTruthOutput:
        result = self.graph.invoke(
            {"submission": submission},
            config={
                "callbacks": self.callbacks,
                "run_name": "groundtruth_pipeline",
                "metadata": {
                    "langfuse_session_id": "groundtruth-session",
                    "langfuse_user_id": "pm-demo-user",
                },
            },
        )
        return GroundTruthOutput(
            intake=submission.model_dump(),
            insight=result["insight"].model_dump(),
            executable_spec=result["executable_spec"].model_dump(),
            high_level_design=result["high_level_design"].model_dump(),
            prd=result["prd"].model_dump(),
        )

    @staticmethod
    def _add_bullet_list(blocks: list[str], items: list[str]) -> None:
        if not items:
            blocks.append("- None provided")
            return
        for item in items:
            blocks.append(f"- {item}")

    @staticmethod
    def to_markdown(output: GroundTruthOutput) -> str:
        blocks = [
            f"# GroundTruth Output: {output.insight.title}",
            "",
            "## Intake",
            f"- **Type:** {output.intake.intake_type.value.replace('_', ' ').title()}",
            f"- **Submitted at:** {output.intake.submitted_at}",
            "",
            "## 1. Insight Summary",
            f"**Title:** {output.insight.title}",
            "",
            "**Problem Statement**",
            output.insight.problem_statement,
            "",
            "**Business Impact**",
            output.insight.business_impact,
            "",
            "**User Segments**",
        ]

        GroundTruthEngine._add_bullet_list(blocks, output.insight.user_segments)
        blocks.extend([
            "",
            "**Assumptions**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.insight.assumptions)
        blocks.extend([
            "",
            "**Risks**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.insight.risks)

        blocks.extend([
            "",
            "## 2. Executable Specification",
            "**Summary**",
            output.executable_spec.summary,
            "",
            "**User Stories**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.executable_spec.user_stories)
        blocks.extend([
            "",
            "**Acceptance Criteria**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.executable_spec.acceptance_criteria)
        blocks.extend([
            "",
            "**Test Scenarios**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.executable_spec.test_scenarios)
        blocks.extend([
            "",
            "**Non-Functional Requirements**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.executable_spec.non_functional_requirements)

        blocks.extend([
            "",
            "## 3. High-Level Design",
            "**Architecture Overview**",
            output.high_level_design.architecture_overview,
            "",
            "**Components**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.high_level_design.components)
        blocks.extend([
            "",
            "**Interfaces**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.high_level_design.interfaces)
        blocks.extend([
            "",
            "**Data Flow**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.high_level_design.data_flow)
        blocks.extend([
            "",
            "**Observability**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.high_level_design.observability)

        blocks.extend([
            "",
            "## 4. Product Requirements Document (PRD)",
            "**Objective**",
            output.prd.objective,
            "",
            "**Success Metrics**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.prd.success_metrics)
        blocks.extend([
            "",
            "**In Scope**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.prd.scope_in)
        blocks.extend([
            "",
            "**Out of Scope**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.prd.scope_out)
        blocks.extend([
            "",
            "**Rollout Notes**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.prd.rollout_notes)
        blocks.extend([
            "",
            "**Open Questions**",
        ])
        GroundTruthEngine._add_bullet_list(blocks, output.prd.open_questions)

        return "\n".join(blocks)
