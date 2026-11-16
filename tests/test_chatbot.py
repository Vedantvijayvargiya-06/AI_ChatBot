"""Automated tests for Smart Study Buddy.

These tests do NOT need a real API key or internet: the Gemini model is replaced by
a fake chat model so we can verify the LangChain wiring and error handling offline.
Live Gemini testing is done with tests/run_live_questions.py (needs your key).

Run:  python -m pytest -v
"""

import os
import sys

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import assistant  # noqa: E402
from chatbot.assistant import StudyBuddyError, ask_question, build_chain, classify_error, get_api_key  # noqa: E402

QUESTIONS = [
    "What is Python and why is it used?",
    "Explain Object-Oriented Programming with an example.",
    "What is Machine Learning?",
    "Explain the difference between AI and ML.",
    "What is a database and why do we need it?",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


# --- Five test questions through the real LCEL chain (fake model) ----------
@pytest.mark.parametrize("question", QUESTIONS)
def test_five_questions_flow_through_chain(question):
    fake = FakeListChatModel(responses=[f"Answer to: {question}"])
    chain = build_chain(llm=fake)
    assert ask_question(question, chain=chain) == f"Answer to: {question}"


def test_prompt_template_injects_question_and_history():
    prompt = assistant.get_prompt()
    assert set(prompt.input_variables) >= {"question"}
    msgs = prompt.invoke({"question": "What is DBMS?", "history": [("human", "hi"), ("ai", "hello")]}).to_messages()
    assert msgs[0].type == "system"
    assert [m.type for m in msgs[1:]] == ["human", "ai", "human"]
    assert msgs[-1].content == "What is DBMS?"


def test_history_is_sent_to_the_model():
    seen = {}

    class Spy(FakeListChatModel):
        def _generate(self, messages, *a, **k):
            seen["messages"] = messages
            return super()._generate(messages, *a, **k)

    chain = build_chain(llm=Spy(responses=["ok"]))
    history = [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "A language."}]
    ask_question("Give an example", history, chain=chain)
    contents = [m.content for m in seen["messages"]]
    assert "What is Python?" in contents and "A language." in contents and contents[-1] == "Give an example"
    assert isinstance(seen["messages"][-1], HumanMessage)


# --- Empty input ------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None])
def test_empty_input_rejected(bad):
    with pytest.raises(StudyBuddyError) as e:
        ask_question(bad)
    assert e.value.kind == "empty"


def test_too_long_question_rejected():
    with pytest.raises(StudyBuddyError):
        ask_question("a" * 5000)


# --- Missing / placeholder API key -----------------------------------------
def test_missing_api_key(monkeypatch):
    monkeypatch.setattr(assistant, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(StudyBuddyError) as e:
        get_api_key()
    assert e.value.kind == "missing_key"
    with pytest.raises(StudyBuddyError) as e2:
        ask_question("What is Python?")
    assert e2.value.kind == "missing_key"


def test_placeholder_key_treated_as_missing(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "your_actual_gemini_api_key")
    with pytest.raises(StudyBuddyError) as e:
        get_api_key()
    assert e.value.kind == "missing_key"


def test_key_is_not_in_error_messages(monkeypatch):
    secret = "AIzaSyFAKE-SECRET-KEY-1234567890"
    monkeypatch.setenv("GOOGLE_API_KEY", secret)

    class Boom(Exception):
        pass

    class FailingChain:
        def invoke(self, _):
            raise Boom(f"request failed for key {secret}")

    with pytest.raises(StudyBuddyError) as e:
        ask_question("hello", chain=FailingChain())
    assert secret not in e.value.message


# --- Error classification (invalid key, quota, network, server) -------------
class FakeAPIError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@pytest.mark.parametrize(
    "exc, kind",
    [
        (FakeAPIError(400, "API key not valid. Please pass a valid API key. API_KEY_INVALID"), "invalid_key"),
        (FakeAPIError(403, "PERMISSION_DENIED"), "invalid_key"),
        (FakeAPIError(429, "RESOURCE_EXHAUSTED: quota exceeded"), "rate_limit"),
        (FakeAPIError(404, "models/x is not found"), "model"),
        (FakeAPIError(503, "UNAVAILABLE: model overloaded"), "server"),
        (ConnectionError("Connection refused"), "network"),
        (TimeoutError("timed out"), "network"),
        (RuntimeError("[Errno -2] getaddrinfo failed"), "network"),
        (ValueError("totally unexpected"), "unknown"),
    ],
)
def test_error_classification(exc, kind):
    assert classify_error(exc).kind == kind


def test_ask_question_wraps_provider_errors_without_traceback():
    class FailingChain:
        def invoke(self, _):
            raise FakeAPIError(429, "RESOURCE_EXHAUSTED")

    with pytest.raises(StudyBuddyError) as e:
        ask_question("What is AI?", chain=FailingChain())
    assert e.value.kind == "rate_limit"
    assert "Traceback" not in e.value.message


def test_empty_model_answer_handled():
    chain = build_chain(llm=FakeListChatModel(responses=["   "]))
    with pytest.raises(StudyBuddyError):
        ask_question("What is AI?", chain=chain)


def test_model_fallback_on_not_found(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key-for-test")
    calls = []

    def fake_build(llm=None, model_name=None):
        calls.append(model_name)

        class C:
            def invoke(self, _):
                if len(calls) == 1:
                    raise FakeAPIError(404, "model not_found")
                return "fallback worked"

        return C()

    monkeypatch.setattr(assistant, "build_chain", fake_build)
    assert ask_question("What is AI?") == "fallback worked"
    assert len(calls) == 2 and calls[0] != calls[1]


def test_model_fallback_on_server_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key-for-test")
    calls = []

    def fake_build(llm=None, model_name=None):
        calls.append(model_name)

        class C:
            def invoke(self, _):
                if len(calls) == 1:
                    raise FakeAPIError(503, "UNAVAILABLE")
                return "second model worked"

        return C()

    monkeypatch.setattr(assistant, "build_chain", fake_build)
    assert ask_question("What is AI?") == "second model worked"
