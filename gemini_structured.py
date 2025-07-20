import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Step 1: Define the schema for structured output
class WeatherInfo(BaseModel):
    location: str = Field(..., description="City name")
    temp_c: float = Field(..., description="Temperature in Celsius")
    condition: str = Field(..., description="Weather condition")

# Step 2: Load Gemini API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Step 3: Create Gemini-compatible OpenAI client
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Step 4: Main logic
def main():
    print("⏳ Requesting structured weather data from Gemini...")

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You respond only with JSON matching the WeatherInfo schema."},
            {"role": "user", "content": "Give me the current weather in Tokyo in structured format."}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "WeatherInfo",
                "schema": WeatherInfo.model_json_schema(),
                "strict": True
            }
        }
    )

    raw_output = response.choices[0].message.content
    print("\n📦 Raw model output:\n", raw_output)

    try:
        # ✅ Corrected for Pydantic v2+
        parsed = WeatherInfo.model_validate_json(raw_output)
        print("\n✅ Parsed WeatherInfo Object:")
        print(f"📍 Location: {parsed.location}")
        print(f"🌡️ Temp: {parsed.temp_c}°C")
        print(f"⛅ Condition: {parsed.condition}")
    except ValidationError as e:
        print("\n❌ Validation failed:")
        print(e)

# Step 5: Run the script
if __name__ == "__main__":
    main()
