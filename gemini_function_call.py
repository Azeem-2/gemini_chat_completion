import os
import json
from dotenv import load_dotenv
from openai import OpenAI

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

# --------------------🌤️ Simulated Tool Function --------------------
def get_weather(location: str) -> dict:
    """Mock function to simulate weather data retrieval."""
    return {
        "location": location,
        "temp": "25°C",
        "condition": "Sunny"
    }

# --------------------🛠️ Define Tool Schema --------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather in a location",
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

# --------------------💬 Main Chat Completion Flow --------------------
def main():
    print("🌐 Asking Gemini for weather in Rawalpindi...\n")

    messages = [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What's the weather like in Rawalpindi today?"}
    ]

    # Step 1: Ask Gemini (initial query with tool awareness)
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_msg = response.choices[0].message
    tool_calls = getattr(assistant_msg, "tool_calls", [])

    if not tool_calls:
        print("🤖 Gemini responded without tool usage:\n")
        print(assistant_msg.content)
        return

    # Step 2: Execute the tool function based on the tool call
    tool_call = tool_calls[0]
    call_id = tool_call.id or "tool_call_1"
    args = json.loads(tool_call.function.arguments)
    tool_result = get_weather(args["location"])

    # Step 3: Append assistant’s tool call and tool response
    messages.append({
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
        ]
    })
    messages.append({
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_call.function.name,
        "content": json.dumps(tool_result)
    })

    # Step 4: Final response from Gemini based on tool output
    final_response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools
    )

    print("✅ Final Response from Gemini:\n")
    print(final_response.choices[0].message.content)

# --------------------🚀 Entry Point --------------------
if __name__ == "__main__":
    main()
