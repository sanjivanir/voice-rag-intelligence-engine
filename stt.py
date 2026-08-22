import os
import requests

def transcribe_audio(file_bytes, filename="test.mp3"):
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        print("SARVAM_API_KEY environment variable is missing.")
        return None

    url = "https://api.sarvam.ai/speech-to-text"
    
    headers = {
        "api-subscription-key": api_key
    }
    
    data = {
        "model": "saaras:v3",
        "language_code": "en-IN",
        "with_timestamps": "false"
    }

    # Pass audio bytes directly with explicit filename and mime type
    files = [
        ("file", (filename, file_bytes, "audio/mpeg"))
    ]

    try:
        response = requests.post(url, headers=headers, data=data, files=files, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            transcript = res_json.get("transcript", "").strip()
            return transcript if transcript else None
        else:
            print(f"Sarvam API Error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        print(f"STT Exception: {str(e)}")
        return None