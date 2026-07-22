from app.models.schemas import IntakeSubmission


def test_intake_submission_accepts_basic_text():
    item = IntakeSubmission(
        intake_type="feature_request",
        raw_text="Project managers want release note approval workflows in Slack.",
    )
    assert item.intake_type == "feature_request"
    assert "Slack" in item.raw_text
