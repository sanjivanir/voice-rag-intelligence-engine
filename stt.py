import os
import requests

def transcribe_audio(file_bytes, filename="audio.wav"):
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        print("SARVAM_API_KEY environment variable is missing.")
        return None

    url = "https://api.sarvam.ai/speech-to-text"
    
    headers = {
        "api-subscription-key": api_key.strip()
    }
    
    # Standard payload for Sarvam STT API
    data = {
        "model": "saaras:v3",
        "language_code": "en-IN",
        "with_timestamps": "false"
    }

    # Determine MIME type dynamically based on file extension
    mime_type = "audio/wav"
    if filename.endswith(".mp3"):
        mime_type = "audio/mpeg"
    elif filename.endswith(".m4a"):
        mime_type = "audio/m4a"

    files = {
        "file": (filename, file_bytes, mime_type)
    }

    try:
        response = requests.post(url, headers=headers, data=data, files=files, timeout=20)
        
        if response.status_code == 200:
            res_json = response.json()
            transcript = res_json.get("transcript", "").strip()
            return transcript if transcript else None
        else:
            print(f"Sarvam API Status {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"STT Exception: {str(e)}")
        return None