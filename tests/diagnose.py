"""Diagnose Gemini connection problems. Run:  python tests\\diagnose.py
Prints the REAL error (key hidden) and the models your key can use."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
key = (os.getenv("GOOGLE_API_KEY") or "").strip()
if not key or key.startswith("your_"):
    sys.exit("No real GOOGLE_API_KEY in .env")
from google import genai
client = genai.Client(api_key=key)
print("== Models available to your key (generateContent) ==")
names = []
try:
    for m in client.models.list():
        if "generateContent" in (getattr(m, "supported_actions", None) or []):
            n = m.name.replace("models/", ""); names.append(n); print(" ", n)
except Exception as e:
    print("Could not list models:", type(e).__name__, str(e).replace(key, "***")[:400])
print("\n== Trying each 'flash' model with a tiny question ==")
for n in [x for x in names if "flash" in x and "tts" not in x and "image" not in x and "live" not in x][:8]:
    try:
        r = client.models.generate_content(model=n, contents="Say OK")
        print(f"  WORKS  {n}: {r.text.strip()[:30]}")
    except Exception as e:
        print(f"  FAILS  {n}: {type(e).__name__}: {str(e).replace(key, '***')[:200]}")
