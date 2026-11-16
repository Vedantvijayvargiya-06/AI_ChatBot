"""Smart Study Buddy - LangChain + Gemini chatbot package."""

from .assistant import (
    StudyBuddyError,
    ask_question,
    build_chain,
    get_api_key,
    get_prompt,
)

__all__ = ["StudyBuddyError", "ask_question", "build_chain", "get_api_key", "get_prompt"]
