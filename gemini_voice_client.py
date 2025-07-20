import os
import openai
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
from multiprocessing import Process, Queue

# Global voice setup
VOICE_ID = None

def choose_voice() -> str:
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    
    print("🗣️ Available Voices:")
    for idx, voice in enumerate(voices):
        print(f"{idx}: {voice.name} - {voice.languages} - {voice.id}")

    while True:
        try:
            chosen_index = int(input("🎚️ Choose voice: 0 for David, 1 for Zira: "))
            if chosen_index in [0, 1]:
                return voices[chosen_index].id
            else:
                print("❌ Invalid input. Choose 0 or 1.")
        except ValueError:
            print("❌ Please enter a number.")

def speak_text(text: str, voice_id: str):
    tts = pyttsx3.init()
    tts.setProperty("rate", 170)
    tts.setProperty("voice", voice_id)
    tts.say(text)
    tts.runAndWait()
    tts.stop()

def main():
    global VOICE_ID

    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("❌ GEMINI_API_KEY not found in .env")

    VOICE_ID = choose_voice()

    # Initialize Gemini client
    client = openai.OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("\n🎤 Gemini Voice Assistant (say 'exit' to stop)\n")

    while True:
        with mic as source:
            print("👂 Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            user_input = recognizer.recognize_google(audio)
            print(f"👤 You: {user_input}")

            if "exit" in user_input.lower():
                print("👋 Exiting...")
                break

            # Send to Gemini
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant."},
                    {"role": "user", "content": user_input}
                ]
            )

            reply = response.choices[0].message.content
            print(f"🤖 Gemini: {reply}")

            # Use a separate process for TTS with selected voice
            p = Process(target=speak_text, args=(reply, VOICE_ID))
            p.start()
            p.join()

        except sr.UnknownValueError:
            print("⚠️ Could not understand audio.")
        except Exception as e:
            print("❌ Error:", e)

if __name__ == "__main__":
    main()
