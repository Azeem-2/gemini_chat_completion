# gemini_basic_chat.py

import os
from openai import OpenAI
from dotenv import load_dotenv

# --------------------🔐 Load API Key --------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

# --------------------🤖 Initialize Gemini Client --------------------
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --------------------💬 Run Basic Chat Completion --------------------
def main():
    print("🧠 Asking Gemini a question...\n")

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "Explain how AI works in simple terms."}
        ]
    )

    message = response.choices[0].message.content
    print("💡 Gemini's Response:\n")
    print(message)

# --------------------🚀 Entry Point --------------------
if __name__ == "__main__":
    main()
