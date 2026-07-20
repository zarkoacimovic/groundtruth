from __future__ import annotations

import os

import streamlit as st

from app.agents.workflow import GroundTruthEngine
from app.integrations.placeholders import planned_integrations
from app.models.schemas import IntakeSubmission, IntakeType
from app.utils.config import get_model_name, tracing_enabled


st.set_page_config(page_title="GroundTruth MVP", page_icon="🧭", layout="wide")


@st.cache_resource
def get_engine() -> GroundTruthEngine:
    return GroundTruthEngine()


def render_form(label: str, intake_type: IntakeType, placeholder: str):
    with st.form(key=f"form_{intake_type.value}"):
        text = st.text_area(
            label,
            height=220,
            placeholder=placeholder,
        )
        submitted = st.form_submit_button("Generate artifacts")
        if submitted:
            if not text.strip():
                st.warning("Please enter some product context.")
                return
            with st.spinner("GroundTruth is generating executable specs, HLD, and PRD..."):
                engine = get_engine()
                result = engine.run(
                    IntakeSubmission(intake_type=intake_type, raw_text=text.strip())
                )
                st.session_state["last_markdown"] = engine.to_markdown(result)
                st.success("Artifacts generated.")
                show_output(result)


def show_output(result):
    st.subheader(result.insight.title)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Insight")
        st.json(result.insight.model_dump())
    with col2:
        st.markdown("### Executable Spec")
        st.json(result.executable_spec.model_dump())
    with col3:
        st.markdown("### High-Level Design")
        st.json(result.high_level_design.model_dump())

    st.markdown("### PRD")
    st.json(result.prd.model_dump())

    st.download_button(
        "Download markdown",
        st.session_state.get("last_markdown", ""),
        file_name="groundtruth_output.md",
        mime="text/markdown",
    )


def sidebar():
    st.sidebar.title("GroundTruth MVP")
    st.sidebar.caption("LangChain + LangGraph + LangSmith + Langfuse")
    st.sidebar.markdown(f"**Model**: `{get_model_name()}`")
    st.sidebar.markdown(f"**LangSmith tracing**: `{tracing_enabled()}`")
    st.sidebar.markdown(f"**Langfuse configured**: `{bool(os.getenv('LANGFUSE_PUBLIC_KEY'))}`")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Future integrations")
    for item in planned_integrations():
        st.sidebar.markdown(f"- **{item.name}** — {item.status}: {item.notes}")


def home():
    st.title("🧭 GroundTruth")
    st.markdown(
        "A simple PM intake-to-artifacts MVP. Paste raw product text into one of four forms and generate grounded outputs: executable specifications, high-level design, and PRD."
    )

    tabs = st.tabs(
        [
            "Service Requests",
            "Customer Found Bugs",
            "New Feature Requests",
            "Competitor Insights",
        ]
    )

    with tabs[0]:
        render_form(
            "Service request input",
            IntakeType.SERVICE_REQUEST,
            "Example: Enterprise customers report long onboarding wait times and want self-service workspace creation.",
        )
    with tabs[1]:
        render_form(
            "Bug report input",
            IntakeType.CUSTOMER_BUG,
            "Example: Users cannot submit claims on Safari after attaching more than one image. Support volume is rising.",
        )
    with tabs[2]:
        render_form(
            "Feature request input",
            IntakeType.FEATURE_REQUEST,
            "Example: PMs want Slack approval workflows for release notes and roadmap changes.",
        )
    with tabs[3]:
        render_form(
            "Competitor insight input",
            IntakeType.COMPETITOR_INSIGHT,
            "Example: Competitor X launched AI ticket triage with Jira sync and executive dashboarding.",
        )


sidebar()
home()
