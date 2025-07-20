import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# --------------------🔐 Load API Key --------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing.")

# --------------------🤖 Initialize Client --------------------
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --------------------🌤 Simple Tool --------------------
def get_current_weather(location: str) -> dict:
    print(f"\n📡 [Tool Called] get_current_weather('{location}')")
    result = {
        "location": location,
        "temperature": "26°C",
        "condition": "Sunny"
    }
    print(f"✅ [Tool Result] {result}\n")
    return result

# --------------------🛠 Tool Definition --------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Returns the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

# --------------------💬 Streaming + Tool Flow --------------------
def main():
    print("🧠 Asking Gemini (streaming with tool access)...\n")

    messages = [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What’s the weather in Lahore?"}
    ]

    # Step 1: Stream from Gemini with tool awareness
    stream = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=True
    )

    tool_calls = []
    buffer = ""

    for chunk in stream:
        delta = chunk.choices[0].delta

        # 1. Collect streamed content
        if delta.content:
            print(delta.content, end="", flush=True)
            buffer += delta.content

        # 2. Detect tool call(s)
        if delta.tool_calls:
            tool_calls.extend(delta.tool_calls)

    print("\n\n🛠 Detected tool calls:", len(tool_calls))

    if not tool_calls:
        return  # No tool call — stream ends here

    # Step 2: Handle the tool calls
    messages.append({
        "role": "assistant",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            } for tc in tool_calls
        ]
    })

    for call in tool_calls:
        args = json.loads(call.function.arguments)
        result = get_current_weather(args["location"])

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.function.name,
            "content": json.dumps(result)
        })

    # Step 3: Final streaming reply from Gemini
    print("\n🔁 Gemini is now finalizing its answer...\n")

    final_stream = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools,
        stream=True
    )

    for chunk in final_stream:
        final_delta = chunk.choices[0].delta
        if final_delta.content:
            print(final_delta.content, end="", flush=True)

    print("\n✅ Done.")

# --------------------🚀 Run --------------------
if __name__ == "__main__":
    main()
