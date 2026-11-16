# 🎓 Smart Study Buddy – AI Study Assistant

**Mini Project 8 – AI Question-Answering Chatbot using LangChain and Google Gemini**

Smart Study Buddy is a web-based chatbot that answers students' academic questions in a
simple, structured, beginner-friendly way. It is built with **Python**, **LangChain (LCEL)**,
the **Google Gemini API** and a modern **Streamlit** interface.

---

## 1. Project Overview

A student types a question (for example *"Explain Object-Oriented Programming with an
example"*). The app inserts the question into a LangChain **prompt template**, sends it to
**Gemini**, and shows the AI-generated answer in a chat window. Answers are generated live by
the model – nothing is hard-coded.
## 🔗 GitHub Repository

The complete source code of this project is available on GitHub.

**Repository Link:** [AI_ChatBot – GitHub](https://github.com/Vedantvijayvargiya-06/AI_ChatBot)

## 🚀 Live Demo

Try Smart Study Buddy here:

[🌐 Launch Live Application](https://dumbchatbot.streamlit.app/)

**Developer:** Vedant Vijay

**Project:** Mini Project 8 – AI Question-Answering Chatbot using LangChain

## 2. Objectives

- Use a language model (Gemini) through LangChain.
- Use a LangChain `ChatPromptTemplate` with a dynamic user question.
- Take a question as input and show an AI-generated answer as output.
- Keep the API key safe in a `.env` file.
- Handle errors gracefully and test with at least five questions.

## 3. Features

- Modern chat UI with chat bubbles, gradient header and sidebar
- Beginner-friendly, structured answers with examples (Python, AI/ML, DSA, DBMS, OS, CN, Maths…)
- Conversation memory for the current session (follow-up questions work)
- Sample-question buttons, **Clear chat** button, loading spinner
- Friendly handling of: empty input, missing/invalid API key, rate limits, no internet,
  Gemini server errors, unexpected errors (no stack traces shown to users)
- Automatic fallback to another Gemini model if the first one is not available for your key

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Programming language |
| LangChain (`langchain-core`) | Prompt template + LCEL chain |
| `langchain-google-genai` | `ChatGoogleGenerativeAI` Gemini integration |
| Google Gemini API | Large language model |
| python-dotenv | Load the API key from `.env` |
| Streamlit | Web interface |
| pytest | Automated tests |

## 5. Project Architecture

```
 Student question
        │
        ▼
 ┌──────────────────┐     ┌────────────────────────┐     ┌──────────────────┐
 │ ChatPromptTemplate│ ──▶ │ ChatGoogleGenerativeAI │ ──▶ │ StrOutputParser  │ ──▶ Answer
 │ (system + history │     │ (Gemini model)         │     │ (AI message→text)│
 │  + {question})    │     └────────────────────────┘     └──────────────────┘
 └──────────────────┘
        prompt | llm | parser      ← LangChain Expression Language (LCEL)
```

## 6. Folder Structure

```
langchain_project/
├── app.py                  # Streamlit user interface
├── .env                    # YOUR API key (never share / commit)
├── .env.example            # Template for .env
├── .gitignore              # Keeps .env out of Git
├── requirements.txt        # Dependencies
├── README.md               # This file
├── chatbot/
│   ├── __init__.py
│   └── assistant.py        # LangChain prompt, chain, error handling
├── tests/
│   ├── test_chatbot.py     # Unit tests (offline, fake model)
│   ├── test_ui.py          # Streamlit UI tests (offline)
│   └── run_live_questions.py  # Live test of 5 questions with real Gemini
└── screenshots/
    └── README.txt          # Put your screenshots here
```

## 7. Installation (Windows CMD)

Install **Python 3.10 or newer** from python.org (tick *"Add Python to PATH"*).

```cmd
cd path\to\langchain_project

:: 1. Create a virtual environment
python -m venv venv

:: 2. Activate it
venv\Scripts\activate

:: 3. Install dependencies
pip install -r requirements.txt
```

(PowerShell: use `venv\Scripts\Activate.ps1`. If blocked, run
`Set-ExecutionPolicy -Scope Process Bypass` first.)

## 8. Get a free Gemini API key

1. Open <https://aistudio.google.com/apikey> and sign in with a Google account.
2. Click **Create API key** and copy it.
3. Open the `.env` file in the project folder and replace the placeholder:

```
GOOGLE_API_KEY=AIza...your_real_key...
```

No quotes, no spaces. Never share this key or upload `.env` to GitHub.

> Optional: add `GEMINI_MODEL=<model name>` to `.env` to force a specific model from
> Google AI Studio. By default the app tries `gemini-3.8-flash`, then `gemini-3.5-flash`,
> then `gemini-3-flash-preview`.

## 9. Run the application

```cmd
venv\Scripts\activate
streamlit run app.py
```

The browser opens at <http://localhost:8501>. Press `Ctrl + C` in CMD to stop.

## 10. How the LangChain Pipeline Works

Defined in `chatbot/assistant.py`:

1. **`ChatPromptTemplate.from_messages`** builds the prompt: a *system* message (rules for
   a friendly study assistant), a `MessagesPlaceholder("history")` for earlier chat turns,
   and a *human* message `{question}` that is filled in dynamically.
2. **`ChatGoogleGenerativeAI`** sends the prompt to Gemini using the key from `.env`.
3. **`StrOutputParser`** converts the model's reply into plain text.
4. The three parts are joined with the LCEL pipe operator:
   `chain = prompt | llm | StrOutputParser()` and run with `chain.invoke({...})`.

## 11. Testing

```cmd
:: Offline tests (no key or internet needed) – 32 tests
python -m pytest -v

:: Live test with the real Gemini API (needs your key + internet)
python tests/run_live_questions.py
```

### Test results recorded while building this project

| Test | How executed | Result |
|---|---|---|
| 5 sample questions through the LCEL chain | pytest, **fake model** (checks prompt → model → parser wiring) | ✅ 5/5 passed |
| Prompt template inserts question + history | pytest | ✅ passed |
| Empty / whitespace / `None` input | pytest + Streamlit AppTest | ✅ passed |
| Missing API key | pytest + Streamlit AppTest | ✅ passed |
| Invalid-key / rate-limit / server / model-not-found error mapping | pytest with simulated provider errors | ✅ passed |
| Network failure | pytest **and** real Gemini client with no reachable network | ✅ friendly message, no key leaked |
| API key never appears in error text | pytest | ✅ passed |
| Clear-chat button, error display, chat roles | Streamlit AppTest | ✅ passed |
| `streamlit run app.py` starts and serves the page | real server, HTTP 200 | ✅ passed |
| Live answers from real Gemini for the 5 questions | `run_live_questions.py` | ✅ passed  |
| Real invalid-key response from Google | live | ✅ passed|

Total automated: **32 passed**.

### Sample questions and expected behaviour

| # | Question | Expected behaviour |
|---|---|---|
| 1 | What is Python and why is it used? | Definition, uses (web, AI, automation), short example |
| 2 | Explain Object-Oriented Programming with an example. | Classes/objects, 4 pillars, Python code example |
| 3 | What is Machine Learning? | Simple definition, types, real-life example |
| 4 | Explain the difference between AI and ML. | Comparison of scope/relationship with examples |
| 5 | What is a database and why do we need it? | Definition, DBMS role, examples (student records) |

## 12. Error Handling

| Situation | What the user sees |
|---|---|
| Empty question | "Please type a question before sending." |
| Missing API key | Instructions to create `.env` and get a free key |
| Invalid API key | "Your Gemini API key was rejected…" |
| Rate limit / quota | "Rate limit or free quota reached. Please wait a minute…" |
| No internet | "Could not reach the Gemini service. Check your connection…" |
| Gemini server busy | "Service temporarily busy. Try again shortly." |
| Anything else | Generic safe message (no stack trace, no secrets) |

## 13. Future Improvements

- Upload PDF notes and answer from them (RAG with embeddings)
- Streaming (word-by-word) answers
- Quiz / flashcard generator and study-plan mode
- Save chat history to a database, user login
- Voice input and multi-language support

---



*Made with Python, LangChain and Google Gemini.*
