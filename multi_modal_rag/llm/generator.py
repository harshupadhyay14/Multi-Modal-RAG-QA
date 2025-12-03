import os
from dotenv import load_dotenv
from groq import Groq, GroqError

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment (.env)")

client = Groq(api_key=GROQ_KEY)

# Primary and fallback models (change as needed)
PRIMARY_MODEL = "llama-3.3-70b-versatile"   # high-quality
FALLBACK_MODEL = "llama-3.1-8b-instant"     # faster / cheaper fallback


def _extract_message_text(choice) -> str:
    """
    Safely extract message text from a Groq choice object.
    Handles different SDK return types:
      - choice.message may be an object with `.content` attribute
      - or a dict-like object with ['content']
      - or nested dict structure.
    """
    try:
        msg = getattr(choice, "message", None) or choice.get("message", None)
    except Exception:
        # choice isn't dict-like and getattr failed — convert to string
        return str(choice)

    # If the message is an object with attribute 'content' or 'value' etc.
    if msg is None:
        return str(choice)

    # If msg is an object (ChatCompletionMessage) with .content
    if hasattr(msg, "content"):
        try:
            return msg.content
        except Exception:
            pass

    # If msg has a 'get' method (dict-like)
    if hasattr(msg, "get"):
        for key in ("content", "text", "message"):
            if msg.get(key) is not None:
                return msg.get(key)

    # If msg can be turned into dict
    try:
        m = dict(msg)
        for key in ("content", "text", "message"):
            if key in m:
                return m[key]
    except Exception:
        pass

    # Fallback to string representation
    return str(msg)


def _call_groq(model: str, prompt: str, temperature: float = 0.0):
    """
    Call Groq chat completions and return the extracted string.
    Raises exceptions for upstream errors; caller will handle fallback.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a document QA assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    # response.choices is expected; choose first non-empty choice
    for choice in getattr(response, "choices", []) or response.get("choices", []):
        text = _extract_message_text(choice)
        if text is not None and str(text).strip():
            return str(text)

    # If no text found, return empty string
    return ""


def generate_answer(context_items, question, temperature: float = 0.0) -> str:
    """
    Generate an answer using Groq. Try PRIMARY_MODEL first; on decommission or
    model errors, try FALLBACK_MODEL. Returns a safe string for the UI.
    """

    # Build compact context (truncate large chunks)
    context = ""
    for c in context_items:
        snippet = c.get("text", "")
        if len(snippet) > 2000:
            snippet = snippet[:2000] + "..."
        context += f"[Page {c.get('page','?')}] {snippet}\n\n"

    prompt = f"""
You are a helpful QA assistant. Use ONLY the provided context to answer the question.
If the document does not contain the answer, respond: "The document does not contain this information."

Context:
{context}

Question: {question}

Answer concisely and include page citations like (Page X).
"""

    # Try primary model first
    try:
        return _call_groq(PRIMARY_MODEL, prompt, temperature)
    except GroqError as ge:
        # GroqError is the SDK's structured error type
        err_text = ""
        try:
            err_text = ge.response.json() if getattr(ge, "response", None) else str(ge)
        except Exception:
            err_text = str(ge)
        print("Groq Error with primary model:", err_text)

        # If model decommissioned or invalid request, try fallback
        if "model_decommissioned" in str(err_text).lower() or "model" in str(err_text).lower():
            try:
                return _call_groq(FALLBACK_MODEL, prompt, temperature)
            except Exception as e2:
                print("Groq Error with fallback model:", str(e2))
                return f"[LLM error] Could not generate answer (models tried: {PRIMARY_MODEL}, {FALLBACK_MODEL}). Reason: {e2}"
        else:
            return f"[LLM error] Could not generate answer (model: {PRIMARY_MODEL}). Reason: {err_text}"

    except Exception as e:
        # Generic exception (network, parsing, etc.)
        print("Groq unexpected error:", str(e))
        # Try fallback model once
        try:
            return _call_groq(FALLBACK_MODEL, prompt, temperature)
        except Exception as e2:
            print("Groq fallback unexpected error:", str(e2))
            return f"[LLM error] Could not generate answer (models tried: {PRIMARY_MODEL}, {FALLBACK_MODEL}). Reason: {e2}"
