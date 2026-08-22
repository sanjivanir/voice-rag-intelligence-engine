import os
import requests

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

def transcribe_audio(file_bytes, filename="audio.mp3"):
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY environment variable is missing.")

    url = "https://api.sarvam.ai/speech-to-text"
    
    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }
    
    # Explicitly set language_code along with model_name
    data = {
        "model": "saaras:v3",
        "language_code": "en-IN",  # Mandatory parameter for Sarvam STT
        "with_timestamps": "false"
    }

    files = {
        "file": (filename, file_bytes, "audio/mpeg")
    }

    response = requests.post(url, headers=headers, data=data, files=files)
    
    if response.status_code == 200:
        res_json = response.json()
        transcript = res_json.get("transcript", "").strip()
        if not transcript:
            return None
        return transcript
    else:
        print(f"Sarvam API Error: {response.status_code} - {response.text}")
        return None