import os
import openai
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
from multiprocessing import Process

def speak_text(text: str):
    tts = pyttsx3.init()
    tts.setProperty('rate', 170)
    tts.say(text)
    tts.runAndWait()
    tts.stop()

def main():
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("❌ GEMINI_API_KEY not found in .env")

    client = openai.OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("🎤 Gemini Voice Assistant (say 'exit' to stop)\n")

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

            resp = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant."},
                    {"role": "user", "content": user_input}
                ]
            )

            reply = resp.choices[0].message.content
            print(f"🤖 Gemini: {reply}")

            # Spawn TTS in separate process for each response
            p = Process(target=speak_text, args=(reply,))
            p.start()
            p.join()

        except sr.UnknownValueError:
            print("⚠️ Could not understand audio.")
        except Exception as e:
            print("❌ Error:", e)

if __name__ == "__main__":
    main()
