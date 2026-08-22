import os
import requests
from dotenv import load_dotenv
from retriever import retrieve_top_k_from_chunks

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Sanitize keys from extra quotes, whitespace, or hidden newlines
raw_groq = os.getenv("GROQ_API_KEY") or ""
groq_key = raw_groq.strip('"' "' \t\r\n")

raw_sarvam = os.getenv("SARVAM_API_KEY") or ""
sarvam_key = raw_sarvam.strip('"' "' \t\r\n")


def get_active_groq_models():
    """Fetch live available model IDs directly from Groq API."""
    if not groq_key:
        return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {groq_key}"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            models = [m["id"] for m in data.get("data", []) if "whisper" not in m["id"].lower()]
            if models:
                return models
    except Exception:
        pass
    return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]


def speech_to_text(audio_path):
    """Transcribe voice audio using Sarvam AI STT API with dual header fallback."""
    clean_sarvam_key = str(sarvam_key).strip('"' "' \t\r\n")

    if not clean_sarvam_key:
        return "ERROR: Missing SARVAM_API_KEY in environment variables."

    url = "https://api.sarvam.ai/speech-to-text"
    
    # Pass dual headers to satisfy all Sarvam platform authentication types
    headers = {
        "api-subscription-key": clean_sarvam_key,
        "Authorization": f"Bearer {clean_sarvam_key}"
    }
    
    ext = os.path.splitext(audio_path)[1].lower()
    mime_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg"
    }
    file_mime = mime_types.get(ext, "audio/mpeg")
    
    data = {
        "model": "saaras:v3",
        "mode": "transcribe",
        "language_code": "unknown",
        "with_timestamps": "false"
    }
    
    try:
        filename = os.path.basename(audio_path)
        with open(audio_path, "rb") as f:
            files = {"file": (filename, f.read(), file_mime)}
            response = requests.post(url, headers=headers, data=data, files=files, timeout=30)
            
        if response.status_code == 200:
            res_json = response.json()
            transcript = res_json.get("transcript", "").strip()
            return transcript if transcript else "Empty transcript returned by API"
        else:
            return f"Sarvam Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"STT Exception: {str(e)}"


def run_rag_pipeline(user_query, chunks=None):
    """Retrieve context directly using passed dataset chunks and query Groq LLM."""
    clean_groq_key = str(groq_key).strip('"' "' \t\r\n")
    
    if not clean_groq_key:
        return "Error: GROQ_API_KEY is missing or empty in environment configuration."

    # Step 1: Retrieve context directly using cached chunks
    contexts = retrieve_top_k_from_chunks(chunks or [], user_query, k=2)
    
    # Strict Guardrail Check
    if not contexts or len("".join(contexts).strip()) < 10:
        return "Guardrail Refusal: Query is out-of-scope or missing relevant context."
        
    context_text = "\n".join(contexts)
    prompt = (
        f"Context from Database:\n{context_text}\n\n"
        f"Question: {user_query}\n"
        f"Answer directly using ONLY the context provided above. "
        f"Do NOT use external knowledge. If the answer cannot be found in the context, "
        f"reply EXACTLY: 'Guardrail Refusal: Query is out-of-scope or missing relevant context.'"
    )
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {clean_groq_key}",
        "Content-Type": "application/json"
    }
    
    candidate_models = get_active_groq_models()
    errors = []
    
    for model_id in candidate_models:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.0
        }
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=10)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
            else:
                errors.append(f"{model_id}: {res.status_code}")
        except Exception as e:
            errors.append(f"{model_id}: {str(e)}")
            
    return f"Error connecting to Groq API. Details: {'; '.join(errors)}"