import os
import requests

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

def transcribe_audio(file_bytes, filename="audio.mp3"):
    api_key = os.getenv("SARVAM_API_KEY") or SARVAM_API_KEY
    if not api_key:
        raise ValueError("SARVAM_API_KEY environment variable is missing.")

    url = "https://api.sarvam.ai/speech-to-text"
    
    headers = {
        "api-subscription-key": api_key
    }
    
    # Payload parameters for Sarvam STT API
    data = {
        "model": "saaras:v3"
    }

    files = {
        "file": (filename, file_bytes, "audio/mpeg")
    }

    try:
        response = requests.post(url, headers=headers, data=data, files=files)
        if response.status_code == 200:
            res_json = response.json()
            return res_json.get("transcript", "").strip()
        else:
            print(f"Sarvam API Error status {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Request exception: {str(e)}")
        return None