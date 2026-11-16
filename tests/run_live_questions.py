"""LIVE test: sends the 5 demo questions to the real Gemini API.

Needs a valid GOOGLE_API_KEY in .env and internet access.
Run:  python tests/run_live_questions.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import StudyBuddyError, ask_question  # noqa: E402

QUESTIONS = [
    "What is Python and why is it used?",
    "Explain Object-Oriented Programming with an example.",
    "What is Machine Learning?",
    "Explain the difference between AI and ML.",
    "What is a database and why do we need it?",
]

passed = 0
for i, q in enumerate(QUESTIONS, 1):
    print(f"\n[{i}/5] Q: {q}")
    try:
        answer = ask_question(q)
        passed += 1
        print("PASS - answer preview:", answer[:300].replace("\n", " "), "...")
    except StudyBuddyError as e:
        print(f"FAIL ({e.kind}): {e.message}")
print(f"\nResult: {passed}/{len(QUESTIONS)} questions answered by Gemini.")
sys.exit(0 if passed == len(QUESTIONS) else 1)
