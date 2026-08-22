import os
import requests

def transcribe_audio(file_bytes, filename="audio.mp3"):
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        print("SARVAM_API_KEY is missing.")
        return None

    url = "https://api.sarvam.ai/speech-to-text"
    
    headers = {
        "api-subscription-key": api_key.strip()
    }

    # Format parameters explicitly for multipart/form-data
    data = {
        "model": "saaras:v3",
        "language_code": "unknown",  # Auto-detects language (English/Hindi)
        "with_timestamps": "false"
    }

    mime_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"

    files = {
        "file": (filename, file_bytes, mime_type)
    }

    try:
        response = requests.post(url, headers=headers, data=data, files=files, timeout=25)
        
        if response.status_code == 200:
            res_json = response.json()
            return res_json.get("transcript", "").strip() or None
        else:
            print(f"Sarvam API Error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        print(f"STT Exception: {str(e)}")
        return None