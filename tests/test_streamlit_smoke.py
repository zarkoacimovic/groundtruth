from streamlit.testing.v1 import AppTest


def test_app_loads_without_exception():
    at = AppTest.from_file("app/main.py")
    at.run(timeout=60)
    assert not at.exception
