"""Core LangChain logic for Smart Study Buddy.

Pipeline (LangChain Expression Language):

    ChatPromptTemplate  |  ChatGoogleGenerativeAI (Gemini)  |  StrOutputParser
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load variables from the .env file (if present) into the environment.
load_dotenv()

# Models are tried in order. If the first one is unavailable for your key
# (HTTP 404), the next one is used. Override with GEMINI_MODEL in .env.
DEFAULT_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-3-flash-preview")

MAX_HISTORY_MESSAGES = 10  # how many previous messages are sent as context
MAX_QUESTION_LENGTH = 4000

SYSTEM_PROMPT = """You are Smart Study Buddy, a friendly and patient AI study assistant for students.

Rules:
- Give accurate, beginner-friendly answers. Use simple language.
- Structure the answer: a short definition first, then key points, then an example.
- Explain difficult ideas with a real-life analogy or a small example.
- For programming questions, include a short, correct code example in a code block.
- You help with Python, AI, Machine Learning, Data Structures, DBMS, Operating Systems,
  Computer Networks, Mathematics and other academic topics.
- If you are not sure about something, say so honestly instead of guessing.
- Keep answers focused; end with a one-line summary when the answer is long."""


class StudyBuddyError(Exception):
    """An error whose message is safe to show directly to the user."""

    def __init__(self, message: str, kind: str = "unknown"):
        super().__init__(message)
        self.message = message
        # missing_key | invalid_key | rate_limit | network | server | model | empty | unknown
        self.kind = kind


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def get_api_key() -> str:
    """Return the Gemini API key from the environment, or raise a friendly error."""
    key = (os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
    if not key or key in {"your_actual_gemini_api_key", "your_api_key_here"}:
        raise StudyBuddyError(
            "Gemini API key not found. Create a file named .env in the project folder "
            "and add the line GOOGLE_API_KEY=your_key (get a free key at "
            "https://aistudio.google.com/apikey). Then restart the app.",
            kind="missing_key",
        )
    return key


def get_model_names() -> list[str]:
    """Models to try, in order. GEMINI_MODEL (from .env) takes priority."""
    custom = (os.getenv("GEMINI_MODEL") or "").strip()
    if custom:
        return [custom] + [m for m in DEFAULT_MODELS if m != custom]
    return list(DEFAULT_MODELS)


# --------------------------------------------------------------------------- #
# LangChain pieces
# --------------------------------------------------------------------------- #
def get_prompt() -> ChatPromptTemplate:
    """The LangChain prompt template: system rules + chat history + the question."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history", optional=True),
            ("human", "{question}"),
        ]
    )


def build_chain(llm=None, model_name: str | None = None):
    """Build the LCEL chain: prompt | model | output parser.

    `llm` can be injected (used by the unit tests); otherwise Gemini is created.
    """
    if llm is None:
        api_key = get_api_key()
        # Imported here so that importing this module never needs the network.
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model_name or get_model_names()[0],
            google_api_key=api_key,
            temperature=0.4,
            max_retries=1,
            timeout=60,
        )
    return get_prompt() | llm | StrOutputParser()


# --------------------------------------------------------------------------- #
# Error translation
# --------------------------------------------------------------------------- #
def _error_info(exc: BaseException) -> tuple[str, set[int], list[str]]:
    """Collect message text, HTTP-like codes and class names along the exception chain."""
    texts, codes, names = [], set(), []
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        names.append(type(cur).__name__.lower())
        texts.append(str(cur).lower())
        for attr in ("code", "status_code", "status"):
            val = getattr(cur, attr, None)
            if isinstance(val, int):
                codes.add(val)
        resp = getattr(cur, "response", None)
        if resp is not None and isinstance(getattr(resp, "status_code", None), int):
            codes.add(resp.status_code)
        cur = cur.__cause__ or cur.__context__
    return " | ".join(texts), codes, names


def classify_error(exc: BaseException) -> StudyBuddyError:
    """Convert any exception into a StudyBuddyError with a user-friendly message."""
    if isinstance(exc, StudyBuddyError):
        return exc

    text, codes, names = _error_info(exc)
    name_blob = " ".join(names)

    if (
        "api key not valid" in text
        or "api_key_invalid" in text
        or "invalid api key" in text
        or "api key expired" in text
        or codes & {401, 403}
        or "permission_denied" in text
        or "unauthenticated" in text
    ):
        return StudyBuddyError(
            "Your Gemini API key was rejected. Please check GOOGLE_API_KEY in your "
            ".env file (no spaces or quotes) or create a new key at "
            "https://aistudio.google.com/apikey, then restart the app.",
            kind="invalid_key",
        )
    if (
        429 in codes
        or "resource_exhausted" in text
        or "quota" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
    ):
        return StudyBuddyError(
            "Rate limit or free quota reached. Please wait a minute and try again.",
            kind="rate_limit",
        )
    if 404 in codes or "not_found" in text or "is not found" in text or "no longer available" in text:
        return StudyBuddyError(
            "The selected Gemini model is not available for your key. Set GEMINI_MODEL "
            "in your .env file to a model listed in Google AI Studio.",
            kind="model",
        )
    if (
        isinstance(exc, (ConnectionError, TimeoutError, OSError))
        or any(k in name_blob for k in ("connect", "timeout", "network", "dns"))
        or any(
            k in text
            for k in (
                "getaddrinfo", "name resolution", "connection error", "connection refused",
                "connection reset", "timed out", "network is unreachable", "failed to establish",
                "temporary failure", "max retries exceeded", "cannot connect",
                "all connection attempts failed",
            )
        )
    ):
        return StudyBuddyError(
            "Could not reach the Gemini service. Please check your internet connection "
            "and try again.",
            kind="network",
        )
    if (codes & {500, 502, 503, 504}) or "unavailable" in text or "overloaded" in text or "internal" in name_blob:
        return StudyBuddyError(
            "The Gemini service is temporarily busy or unavailable. Please try again shortly.",
            kind="server",
        )
    return StudyBuddyError(
        "Something went wrong while generating the answer. Please try again.",
        kind="unknown",
    )


def _debug(exc: BaseException, model_name: str) -> None:
    """If STUDY_BUDDY_DEBUG=1, print the raw error to the console (key redacted)."""
    if os.getenv("STUDY_BUDDY_DEBUG") == "1":
        msg = str(exc)
        key = os.getenv("GOOGLE_API_KEY") or ""
        if key:
            msg = msg.replace(key, "***")
        print(f"[debug] model={model_name} {type(exc).__name__}: {msg[:600]}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _to_history(messages: Iterable[dict] | None) -> list[tuple[str, str]]:
    """Convert [{'role': 'user'|'assistant', 'content': str}] to LangChain tuples."""
    out: list[tuple[str, str]] = []
    for m in list(messages or [])[-MAX_HISTORY_MESSAGES:]:
        role = "human" if m.get("role") == "user" else "ai"
        content = str(m.get("content", "")).strip()
        if content:
            out.append((role, content))
    return out


def ask_question(question: str, history: Sequence[dict] | None = None, chain=None) -> str:
    """Send a question to Gemini through the LangChain chain and return the answer text.

    Raises StudyBuddyError (with a safe, friendly message) for every failure.
    """
    question = (question or "").strip()
    if not question:
        raise StudyBuddyError("Please type a question before sending.", kind="empty")
    if len(question) > MAX_QUESTION_LENGTH:
        raise StudyBuddyError(
            f"Your question is too long (max {MAX_QUESTION_LENGTH} characters). Please shorten it.",
            kind="empty",
        )

    payload = {"question": question, "history": _to_history(history)}

    try:
        if chain is not None:  # injected chain (used by tests)
            answer = chain.invoke(payload)
        else:
            get_api_key()  # raises a friendly error if the key is missing
            answer, last_error = None, None
            for model_name in get_model_names():
                try:
                    answer = build_chain(model_name=model_name).invoke(payload)
                    break
                except Exception as exc:  # noqa: BLE001 - classified below
                    last_error = classify_error(exc)
                    _debug(exc, model_name)
                    # Try the next model only if this one is missing or overloaded.
                    if last_error.kind not in ("model", "server"):
                        raise last_error from None
            if answer is None:
                raise last_error  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001 - never leak stack traces to the UI
        raise classify_error(exc) from None

    answer = (answer or "").strip()
    if not answer:
        raise StudyBuddyError(
            "The model returned an empty answer. Please rephrase your question and try again.",
            kind="unknown",
        )
    return answer
