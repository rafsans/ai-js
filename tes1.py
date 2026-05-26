from google import genai
from dotenv import load_dotenv
import os

# load .env
load_dotenv()

# ambil API key
api_key = os.getenv("GEMINI_API_KEY")

print("API KEY:", api_key[:10] + "..." if api_key else "NOT FOUND")

try:
    # buat client
    client = genai.Client(api_key=api_key)

    # test request
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in one sentence."
    )

    print("\n====================")
    print("API KEY VALID")
    print("====================")
    print(response.text)

except Exception as e:
    print("\n====================")
    print("API ERROR")
    print("====================")
    print(e)