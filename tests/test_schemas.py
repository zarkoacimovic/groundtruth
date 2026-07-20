from app.models.schemas import IntakeSubmission, IntakeType


def test_intake_submission_accepts_basic_text():
    item = IntakeSubmission(
        intake_type=IntakeType.FEATURE_REQUEST,
        raw_text="Project managers want release note approval workflows in Slack.",
    )
    assert item.intake_type == IntakeType.FEATURE_REQUEST
    assert "Slack" in item.raw_text
