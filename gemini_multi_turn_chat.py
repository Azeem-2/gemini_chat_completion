import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import pytz

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing.")

# Initialize client
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Optional memory file
MEMORY_FILE = "chat_memory.json"

# Define tools
def get_current_weather(location: str) -> dict:
    print(f"📡 [Tool Called] get_current_weather('{location}')")
    return {"location": location, "temperature": "26°C", "condition": "Sunny"}

def get_current_time(city: str) -> dict:
    print(f"📡 [Tool Called] get_current_time('{city}')")
    tz_map = {
        "lahore": "Asia/Karachi",
        "new york": "America/New_York",
        "tokyo": "Asia/Tokyo",
        "london": "Europe/London"
    }
    tz = pytz.timezone(tz_map.get(city.lower(), "UTC"))
    now = datetime.now(tz).strftime("%I:%M %p")
    return {"city": city, "current_time": now}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Returns the weather for a location",
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
            "description": "Returns the current time in a city",
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

# Load memory from disk (optional)
def load_memory() -> list:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [{"role": "system", "content": "You are a helpful assistant."}]

# Save memory to disk
def save_memory(messages: list):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)

# Main interactive loop
def chat_loop():
    messages = load_memory()
    print("💬 Gemini Chat (with multi-turn memory & tools) — type 'exit' to stop\n")

    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in {"exit", "quit"}:
            save_memory(messages)
            print("💾 Chat memory saved. Goodbye!")
            break

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # First request — LLM may call tool(s)
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        assistant_msg = response.choices[0].message
        tool_calls = getattr(assistant_msg, "tool_calls", [])
        messages.append(assistant_msg)

        # If tools were called
        if tool_calls:
            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                result = (
                    get_current_weather(args["location"])
                    if tc.function.name == "get_current_weather"
                    else get_current_time(args["city"])
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(result)
                })

            # Follow-up call to LLM with tool results
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages,
                tools=tools
            )
            assistant_msg = response.choices[0].message
            messages.append(assistant_msg)

        print("🤖 Gemini:", assistant_msg.content)

# Run
if __name__ == "__main__":
    chat_loop()
