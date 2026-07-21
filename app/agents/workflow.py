from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langgraph.graph import END, START, StateGraph

from app.models.schemas import (
    ExecutableSpecification,
    GroundTruthOutput,
    HighLevelDesign,
    InsightSummary,
    IntakeSubmission,
    PRD,
)
from app.prompts.templates import HLD_PROMPT, INSIGHT_PROMPT, PRD_PROMPT, SPEC_PROMPT


class GroundTruthState(TypedDict, total=False):
    submission: IntakeSubmission
    insight: InsightSummary
    executable_spec: ExecutableSpecification
    high_level_design: HighLevelDesign
    prd: PRD


class GroundTruthEngine:
    def __init__(self) -> None:
        model_name = os.getenv("GROUNDTRUTH_MODEL", "gemini-3.5-flash")
        temperature = float(os.getenv("GROUNDTRUTH_TEMPERATURE", "0.2"))

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        )

        workflow = StateGraph(GroundTruthState)
        workflow.add_node("insight", self._create_insight)
        workflow.add_node("spec", self._create_spec)
        workflow.add_node("hld", self._create_hld)
        workflow.add_node("prd", self._create_prd)

        workflow.add_edge(START, "insight")
        workflow.add_edge("insight", "spec")
        workflow.add_edge("spec", "hld")
        workflow.add_edge("hld", "prd")
        workflow.add_edge("prd", END)

        compiled = workflow.compile().with_config({"run_name": "groundtruth_pipeline"})
        self.app = RunnablePassthrough() | compiled

    def _langfuse_callbacks(self) -> List[Any]:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        base_url = os.getenv("LANGFUSE_BASE_URL")

        if not public_key or not secret_key:
            return []

        return [
            LangfuseCallbackHandler(
                public_key=public_key,
                secret_key=secret_key,
                host=base_url,
            )
        ]

    def _submission_data(self, submission: IntakeSubmission) -> Dict[str, Any]:
        if hasattr(submission, "model_dump"):
            return submission.model_dump()
        if isinstance(submission, dict):
            return submission
        return {}

    def _field(self, submission: IntakeSubmission, name: str, default: str = "") -> str:
        try:
            value = getattr(submission, name)
            if value is not None:
                return str(value)
        except AttributeError:
            pass

        data = self._submission_data(submission)
        value = data.get(name, default)
        if value is None:
            return default
        return str(value)

    def _submission_to_text(self, submission: IntakeSubmission) -> str:
        if hasattr(submission, "model_dump_json"):
            return submission.model_dump_json(indent=2)
        return str(submission)

    def _build_invoke_config(
        self,
        submission: IntakeSubmission,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> RunnableConfig:
        intake_type = self._field(submission, "intake_type", "unknown")
        requested_by = self._field(submission, "requested_by", "anonymous-user")
        title = self._field(submission, "title", "untitled-submission")

        effective_session_id = session_id or f"groundtruth-{intake_type}-{uuid4().hex[:12]}"
        effective_user_id = user_id or requested_by or "anonymous-user"

        metadata: Dict[str, Any] = {
            "langfuse_session_id": effective_session_id,
            "langfuse_user_id": effective_user_id,
            "intake_type": intake_type,
            "submission_title": title,
        }

        if tags:
            metadata["langfuse_tags"] = tags

        return {
            "callbacks": self._langfuse_callbacks(),
            "metadata": metadata,
            "run_name": "groundtruth_pipeline",
        }

    def _create_insight(
        self,
        state: GroundTruthState,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, InsightSummary]:
        prompt = ChatPromptTemplate.from_template(INSIGHT_PROMPT)
        chain = prompt | self.llm.with_structured_output(InsightSummary)
        result = chain.invoke(
            {"submission": self._submission_to_text(state["submission"])},
            config=config,
        )
        return {"insight": result}

    def _create_spec(
        self,
        state: GroundTruthState,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, ExecutableSpecification]:
        prompt = ChatPromptTemplate.from_template(SPEC_PROMPT)
        chain = prompt | self.llm.with_structured_output(ExecutableSpecification)
        result = chain.invoke(
            {
                "submission": self._submission_to_text(state["submission"]),
                "insight": state["insight"].model_dump_json(indent=2),
            },
            config=config,
        )
        return {"executable_spec": result}

    def _create_hld(
        self,
        state: GroundTruthState,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, HighLevelDesign]:
        prompt = ChatPromptTemplate.from_template(HLD_PROMPT)
        chain = prompt | self.llm.with_structured_output(HighLevelDesign)
        result = chain.invoke(
            {
                "submission": self._submission_to_text(state["submission"]),
                "insight": state["insight"].model_dump_json(indent=2),
                "spec": state["executable_spec"].model_dump_json(indent=2),
            },
            config=config,
        )
        return {"high_level_design": result}

    def _create_prd(
        self,
        state: GroundTruthState,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, PRD]:
        prompt = ChatPromptTemplate.from_template(PRD_PROMPT)
        chain = prompt | self.llm.with_structured_output(PRD)
        result = chain.invoke(
            {
                "submission": self._submission_to_text(state["submission"]),
                "insight": state["insight"].model_dump_json(indent=2),
                "spec": state["executable_spec"].model_dump_json(indent=2),
                "hld": state["high_level_design"].model_dump_json(indent=2),
            },
            config=config,
        )
        return {"prd": result}

    def run(
        self,
        submission: IntakeSubmission,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> GroundTruthOutput:
        state = self.app.invoke(
            {"submission": submission},
            config=self._build_invoke_config(
                submission=submission,
                session_id=session_id,
                user_id=user_id,
                tags=tags,
            ),
        )

        return GroundTruthOutput(
            submission=state["submission"].model_dump(),
            insight=state["insight"].model_dump(),
            executable_spec=state["executable_spec"].model_dump(),
            high_level_design=state["high_level_design"].model_dump(),
            prd=state["prd"].model_dump(),
        )

    def to_markdown(self, output: GroundTruthOutput) -> str:
        submission = output.submission.model_dump()
        insight = output.insight.model_dump()
        spec = output.executable_spec.model_dump()
        hld = output.high_level_design.model_dump()
        prd = output.prd.model_dump()

        def bullet_list(items: List[str]) -> str:
            if not items:
                return "- None"
            return "\n".join(f"- {item}" for item in items)

        prd_sections = prd.get("sections", [])
        if prd_sections:
            prd_text = "\n\n".join(
                f"### {section.get('title', 'Section')}\n{section.get('content', '').strip()}"
                for section in prd_sections
            )
        else:
            prd_text = prd.get("summary", "No PRD content generated.")

        return f"""# GroundTruth Service Request

## Intake
- **Type:** {submission.get('intake_type', 'N/A')}
- **Title:** {submission.get('title', 'N/A')}
- **Requested by:** {submission.get('requested_by', 'N/A')}
- **Business context:** {submission.get('business_context', 'N/A')}
- **Problem statement:** {submission.get('problem_statement', 'N/A')}
- **Desired outcome:** {submission.get('desired_outcome', 'N/A')}

## Insight Summary
**Summary**
{insight.get('summary', 'N/A')}

**Key themes**
{bullet_list(insight.get('key_themes', []))}

**Risks**
{bullet_list(insight.get('risks', []))}

**Recommendations**
{bullet_list(insight.get('recommendations', []))}

## Executable Specification
**Overview**
{spec.get('summary', 'N/A')}

**Functional requirements**
{bullet_list(spec.get('functional_requirements', []))}

**Non-functional requirements**
{bullet_list(spec.get('non_functional_requirements', []))}

**Acceptance criteria**
{bullet_list(spec.get('acceptance_criteria', []))}

## High-Level Design
**Architecture summary**
{hld.get('overview', 'N/A')}

**Components**
{bullet_list(hld.get('components', []))}

**Data flow**
{bullet_list(hld.get('data_flow', []))}

**Dependencies**
{bullet_list(hld.get('dependencies', []))}

## Product Requirements Document
{prd_text}
"""
