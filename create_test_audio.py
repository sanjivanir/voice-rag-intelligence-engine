from gtts import gTTS

# Dictionary of filenames and test queries
test_files = {
    "test_heart.mp3": "What is the function of the human heart?",
    "test_sun.mp3": "Why is the sun important to Earth?",
    "test_water.mp3": "What is the boiling point of water?",
    "test_guardrail.mp3": "How do I bake a chocolate cake at home?"  # Out-of-scope query
}

for filename, text in test_files.items():
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    print(f"Generated: {filename}")