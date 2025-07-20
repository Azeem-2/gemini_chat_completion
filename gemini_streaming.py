# gemini_streaming.py

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

# --------------------💬 Stream Gemini Response --------------------
def chat_stream(prompt: str):
    print(f"📤 Sending prompt: {prompt}\n")
    print("📥 Streaming Gemini's response...\n")

    # Request with streaming enabled
    stream = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        reasoning_effort="low"  # options: low, medium, high
    )

    # Output response chunk by chunk
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)

    print("\n\n✅ Streaming complete.")

# --------------------🚀 Entry Point --------------------
if __name__ == "__main__":
    chat_stream("Tell me a story about a clever cat.")
