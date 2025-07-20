import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# --------------------🔐 Load API Key --------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing.")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --------------------📦 Final Structured Schema --------------------
class WeatherSummary(BaseModel):
    location: str = Field(..., description="City name")
    summary: str = Field(..., description="Brief weather description")

# --------------------🌦 Tool Function --------------------
def get_weather(location: str) -> dict:
    print(f"\n📡 [Tool Called] get_current_weather('{location}')")
    return {"location": location, "temperature": "26°C", "condition": "Sunny"}

# --------------------🛠 Define Tools --------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Returns the weather for a given city.",
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

# --------------------💬 Full Flow --------------------
def main():
    # Step 1: Ask Gemini with tools only
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Lahore?"}
    ]

    print("🧠 Step 1: Asking Gemini to call tool...\n")
    tool_call_resp = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_msg = tool_call_resp.choices[0].message
    tool_calls = getattr(assistant_msg, "tool_calls", [])

    if not tool_calls:
        print("No tool was called.")
        print(assistant_msg.content)
        return

    # Handle tool result
    tool_call = tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    tool_result = get_weather(args["location"])

    messages.append(assistant_msg)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": json.dumps(tool_result)
    })

    # Step 2: Ask Gemini to format it strictly
    print("📦 Step 2: Ask Gemini to return structured JSON...\n")

    final_response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "WeatherSummary",
                "schema": WeatherSummary.model_json_schema(),
                "strict": True
            }
        }
    )

    content = final_response.choices[0].message.content
    print("📥 Structured JSON Response:\n", content)

    try:
        parsed = WeatherSummary.model_validate_json(content)
        print(f"\n✅ Parsed Output: {parsed.location} — {parsed.summary}")
    except ValidationError as e:
        print("\n❌ Schema validation failed:\n", e)

# --------------------🚀 Run --------------------
if __name__ == "__main__":
    main()
