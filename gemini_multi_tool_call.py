import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import pytz

# --------------------🔐 Load API Key --------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing in .env")

# --------------------🤖 Initialize Gemini Client --------------------
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --------------------🌤 Tool 1: Weather --------------------
def get_current_weather(location: str) -> dict:
    print(f"📡 [Tool Called] get_current_weather(location='{location}')")
    result = {
        "location": location,
        "temperature": "26°C",
        "condition": "Sunny"
    }
    print(f"✅ [Tool Result] {result}\n")
    return result

# --------------------🕒 Tool 2: Time --------------------
def get_current_time(city: str) -> dict:
    print(f"📡 [Tool Called] get_current_time(city='{city}')")
    try:
        timezone_map = {
            "tokyo": "Asia/Tokyo",
            "lahore": "Asia/Karachi"
        }
        tz = pytz.timezone(timezone_map.get(city.lower(), "UTC"))
        now = datetime.now(tz).strftime("%I:%M %p")
        result = {"city": city, "current_time": now}
    except Exception as e:
        result = {"city": city, "error": str(e)}
    print(f"✅ [Tool Result] {result}\n")
    return result

# --------------------🛠 Tool Definitions --------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Returns the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current time in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

# --------------------💬 Multi-Tool Flow --------------------
def main():
    print("🌍 Asking Gemini for weather and time...\n")

    messages = [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": "What is the weather in Lahore and what time is it in Tokyo?"}
    ]

    # Step 1: Gemini generates tool calls
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_msg = response.choices[0].message
    tool_calls = getattr(assistant_msg, "tool_calls", [])

    if not tool_calls:
        print("🤖 Gemini did not use any tools:\n")
        print(assistant_msg.content)
        return

    messages.append(assistant_msg)

    # Step 2: Loop through tool calls and log each one
    for call in tool_calls:
        name = call.function.name
        args = json.loads(call.function.arguments)

        if name == "get_current_weather":
            result = get_current_weather(args["location"])
        elif name == "get_current_time":
            result = get_current_time(args["city"])
        else:
            result = {"error": "Unknown tool"}

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": name,
            "content": json.dumps(result)
        })

    # Step 3: Final response from Gemini
    final_response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools
    )

    print("✅ Gemini’s Final Answer:\n")
    print(final_response.choices[0].message.content)

# --------------------🚀 Run --------------------
if __name__ == "__main__":
    main()
