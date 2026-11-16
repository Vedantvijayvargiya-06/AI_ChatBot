"""UI tests using Streamlit's AppTest (runs app.py headlessly, no browser/API needed)."""
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
APP = os.path.join(ROOT, "app.py")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_app_loads_and_shows_missing_key_warning():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("API key not found" in w.value for w in at.warning)
    assert "Smart Study Buddy" in " ".join(m.value for m in at.markdown)


def test_question_without_key_shows_friendly_error():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("What is Python?").run()
    assert not at.exception
    assert any("API key not found" in e.value for e in at.error)


def test_whitespace_question_shows_warning():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("   ").run()
    assert any("type a question" in w.value for w in at.warning)


def test_clear_button_resets_chat():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("What is AI?").run()
    assert len(at.session_state["messages"]) == 2
    [b for b in at.sidebar.button if "Clear" in b.label][0].click().run()
    assert at.session_state["messages"] == []


def test_successful_answer_is_displayed(monkeypatch):
    import chatbot.assistant as a
    monkeypatch.setattr(a, "ask_question", lambda q, h=None, chain=None: "Python is a language.")
    # app.py imports ask_question by name, so patch the module attribute it binds at import time
    import chatbot
    monkeypatch.setattr(chatbot, "ask_question", a.ask_question)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("What is Python?").run()
    assert not at.exception
    assert [m["role"] for m in at.session_state["messages"]] == ["user", "assistant"]
