from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

# Ensure repo root is on the import path when running on Streamlit Cloud
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from app.agents.workflow import GroundTruthEngine
from app.models.schemas import IntakeSubmission


st.set_page_config(
    page_title="GroundTruth MVP",
    page_icon="🧭",
    layout="wide",
)


INTAKE_CONFIG = {
    "service_request": {
        "label": "Service Requests",
        "help": "Capture customer service requests and convert them into structured delivery artifacts.",
        "problem_label": "Customer request details",
        "outcome_label": "Desired business outcome",
        "example": "Enterprise customer wants faster onboarding, clearer request ownership, and SLA visibility.",
    },
    "customer_bug": {
        "label": "Customer Found Bugs",
        "help": "Capture customer-reported defects, impact, and expected resolution outcome.",
        "problem_label": "Bug description",
        "outcome_label": "Expected fix outcome",
        "example": "Customers report that saving updates sometimes fails after page refresh, causing lost work.",
    },
    "feature_request": {
        "label": "New Feature Requests",
        "help": "Capture requested capabilities and translate them into specifications and product artifacts.",
        "problem_label": "Requested feature / user problem",
        "outcome_label": "Expected value if delivered",
        "example": "Users want saved views and advanced filtering to manage large volumes of records more efficiently.",
    },
    "competitor_insight": {
        "label": "Competitor Insights",
        "help": "Capture competitive observations and turn them into strategic product responses.",
        "problem_label": "Competitive insight",
        "outcome_label": "Desired strategic response",
        "example": "A competitor offers faster onboarding and stronger Slack/Jira integration, creating adoption risk.",
    },
}


def init_session_state() -> None:
    if "engine" not in st.session_state:
        st.session_state.engine = GroundTruthEngine()

    # One stable Langfuse session per Streamlit browser session.
    if "langfuse_session_id" not in st.session_state:
        st.session_state.langfuse_session_id = f"groundtruth-ui-{uuid4().hex[:12]}"

    for intake_type in INTAKE_CONFIG.keys():
        result_key = f"result_{intake_type}"
        markdown_key = f"markdown_{intake_type}"
        if result_key not in st.session_state:
            st.session_state[result_key] = None
        if markdown_key not in st.session_state:
            st.session_state[markdown_key] = None


def get_secret_or_env(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)


def sidebar_status() -> None:
    st.sidebar.title("GroundTruth MVP")
    st.sidebar.caption("LangChain + LangGraph + Gemini + LangSmith + Langfuse")

    google_key = bool(get_secret_or_env("GOOGLE_API_KEY") or get_secret_or_env("GEMINI_API_KEY"))
    langsmith_enabled = get_secret_or_env("LANGSMITH_TRACING", "false").lower() == "true"
    langfuse_enabled = bool(get_secret_or_env("LANGFUSE_PUBLIC_KEY") and get_secret_or_env("LANGFUSE_SECRET_KEY"))
    model_name = get_secret_or_env("GROUNDTRUTH_MODEL", "gemini-3.5-flash")

    st.sidebar.markdown("### Runtime status")
    st.sidebar.write(f"**Model:** {model_name}")
    st.sidebar.write(f"**Gemini key:** {'✅ configured' if google_key else '❌ missing'}")
    st.sidebar.write(f"**LangSmith tracing:** {'✅ enabled' if langsmith_enabled else '⚪ disabled'}")
    st.sidebar.write(f"**Langfuse tracing:** {'✅ enabled' if langfuse_enabled else '⚪ disabled'}")

    st.sidebar.markdown("### Current UI session")
    st.sidebar.code(st.session_state.langfuse_session_id)
    st.sidebar.caption("This session ID is reused across requests in the same browser session and passed to Langfuse.")

    st.sidebar.markdown("### Planned integrations")
    st.sidebar.write("- Slack")
    st.sidebar.write("- ServiceNow")
    st.sidebar.write("- Jira")


def build_submission(
    intake_type: str,
    title: str,
    requested_by: str,
    business_context: str,
    problem_statement: str,
    desired_outcome: str,
) -> IntakeSubmission:
    clean_title = title.strip()
    clean_requested_by = requested_by.strip() or "Anonymous User"
    clean_business_context = business_context.strip()
    clean_problem_statement = problem_statement.strip()
    clean_desired_outcome = desired_outcome.strip()

    raw_text = f"""Title: {clean_title}
Requested by: {clean_requested_by}
Intake type: {intake_type}
Business context: {clean_business_context}
Problem statement: {clean_problem_statement}
Desired outcome: {clean_desired_outcome}"""

    return IntakeSubmission(
        intake_type=intake_type,
        title=clean_title,
        requested_by=clean_requested_by,
        business_context=clean_business_context,
        problem_statement=clean_problem_statement,
        desired_outcome=clean_desired_outcome,
        raw_text=raw_text,
    )


def show_output(intake_type: str) -> None:
    result = st.session_state.get(f"result_{intake_type}")
    markdown = st.session_state.get(f"markdown_{intake_type}")

    if not result:
        return

    st.success("Artifacts generated successfully.")

    submission_tab, insight_tab, spec_tab, hld_tab, prd_tab = st.tabs(
        ["Intake", "Insight", "Executable Spec", "HLD", "PRD"]
    )

    with submission_tab:
        st.json(result.submission.model_dump())

    with insight_tab:
        st.json(result.insight.model_dump())

    with spec_tab:
        st.json(result.executable_spec.model_dump())

    with hld_tab:
        st.json(result.high_level_design.model_dump())

    with prd_tab:
        st.json(result.prd.model_dump())

    if markdown:
        st.download_button(
            label="Download markdown export",
            data=markdown,
            file_name=f"groundtruth_{intake_type}.md",
            mime="text/markdown",
            key=f"download_{intake_type}",
        )


def run_generation(
    intake_type: str,
    title: str,
    requested_by: str,
    business_context: str,
    problem_statement: str,
    desired_outcome: str,
) -> None:
    submission = build_submission(
        intake_type=intake_type,
        title=title,
        requested_by=requested_by,
        business_context=business_context,
        problem_statement=problem_statement,
        desired_outcome=desired_outcome,
    )

    user_id = requested_by.strip() or "anonymous-user"
    session_id = st.session_state.langfuse_session_id

    result = st.session_state.engine.run(
        submission,
        session_id=session_id,
        user_id=user_id,
        tags=["groundtruth", "streamlit", intake_type],
    )

    st.session_state[f"result_{intake_type}"] = result
    st.session_state[f"markdown_{intake_type}"] = st.session_state.engine.to_markdown(result)


def render_form(intake_type: str) -> None:
    cfg = INTAKE_CONFIG[intake_type]
    form_key = f"form_{intake_type}"

    st.subheader(cfg["label"])
    st.caption(cfg["help"])

    with st.form(form_key):
        title = st.text_input(
            "Title",
            placeholder=f"Example: {cfg['label']} intake",
            key=f"title_{intake_type}",
        )
        requested_by = st.text_input(
            "Requested by",
            placeholder="Example: Product Manager / Customer / Support Lead",
            key=f"requested_by_{intake_type}",
        )
        business_context = st.text_area(
            "Business context",
            placeholder="Describe the business context, user segment, process, or account situation.",
            height=120,
            key=f"business_context_{intake_type}",
        )
        problem_statement = st.text_area(
            cfg["problem_label"],
            placeholder=cfg["example"],
            height=150,
            key=f"problem_statement_{intake_type}",
        )
        desired_outcome = st.text_area(
            cfg["outcome_label"],
            placeholder="Describe the desired outcome, expected resolution, or business value.",
            height=120,
            key=f"desired_outcome_{intake_type}",
        )

        submitted = st.form_submit_button("Generate artifacts", use_container_width=True)

    if submitted:
        missing_fields = []
        if not title.strip():
            missing_fields.append("Title")
        if not business_context.strip():
            missing_fields.append("Business context")
        if not problem_statement.strip():
            missing_fields.append(cfg["problem_label"])
        if not desired_outcome.strip():
            missing_fields.append(cfg["outcome_label"])

        if missing_fields:
            st.error("Please complete the following fields: " + ", ".join(missing_fields))
        else:
            with st.spinner("Generating insight, specification, design, and PRD..."):
                run_generation(
                    intake_type=intake_type,
                    title=title,
                    requested_by=requested_by,
                    business_context=business_context,
                    problem_statement=problem_statement,
                    desired_outcome=desired_outcome,
                )

    show_output(intake_type)


def home() -> None:
    st.title("GroundTruth")
    st.write(
        "Turn product and customer inputs into structured artifacts: insight summary, executable specification, high-level design, and PRD."
    )

    st.info(
        "This MVP uses one persistent Langfuse session ID per Streamlit browser session so related runs can be grouped together in observability views."
    )

    tab_labels = [cfg["label"] for cfg in INTAKE_CONFIG.values()]
    tabs = st.tabs(tab_labels)

    for tab, intake_type in zip(tabs, INTAKE_CONFIG.keys()):
        with tab:
            render_form(intake_type)


def main() -> None:
    init_session_state()
    sidebar_status()
    home()


if __name__ == "__main__":
    main()
