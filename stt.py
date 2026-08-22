import os
import requests
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

SARVAM_API_KEY = (os.getenv("SARVAM_API_KEY") or "").strip('"' "' \t\r\n")

def speech_to_text(audio_file_path: str) -> str:
    """
    Transcribes audio using Sarvam AI's Saaras v3 model.
    Supports .wav, .mp3, .m4a, and .flac formats.
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is missing from environment variables!")
        
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found at path: {audio_file_path}")

    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": SARVAM_API_KEY}
    
    with open(audio_file_path, "rb") as audio_file:
        files = {"file": (os.path.basename(audio_file_path), audio_file, "audio/wav")}
        data = {"model": "saaras:v3"}
        
        response = requests.post(url, headers=headers, files=files, data=data)
        
    if response.status_code == 200:
        result = response.json()
        return result.get("transcript", "")
    else:
        raise Exception(f"Sarvam AI STT API Error ({response.status_code}): {response.text}")

if __name__ == "__main__":
    print("STT Module Loaded. Provide an audio file path to test.")