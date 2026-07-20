from __future__ import annotations

import json
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
    from langfuse.callback import CallbackHandler
except Exception:  # pragma: no cover - optional dependency path
    CallbackHandler = None


class GroundTruthState(TypedDict, total=False):
    submission: IntakeSubmission
    insight: InsightSummary
    executable_spec: ExecutableSpecification
    high_level_design: HighLevelDesign
    prd: PRD


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
            config={"callbacks": self.callbacks, "run_name": "groundtruth_pipeline"},
        )
        return GroundTruthOutput(
            intake=submission.model_dump(),
            insight=result["insight"].model_dump(),
            executable_spec=result["executable_spec"].model_dump(),
            high_level_design=result["high_level_design"].model_dump(),
            prd=result["prd"].model_dump(),
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
            "```json",
            json.dumps(output.insight.model_dump(), indent=2),
            "```",
            "",
            "## Executable Specification",
            "```json",
            json.dumps(output.executable_spec.model_dump(), indent=2),
            "```",
            "",
            "## High-Level Design",
            "```json",
            json.dumps(output.high_level_design.model_dump(), indent=2),
            "```",
            "",
            "## PRD",
            "```json",
            json.dumps(output.prd.model_dump(), indent=2),
            "```",
        ]
        return "\n".join(blocks)
