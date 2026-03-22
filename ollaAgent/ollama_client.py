import os

from dotenv import load_dotenv
from ollama import Client

load_dotenv()

client = Client(
    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    headers={
        "CF-Access-Client-Id": os.getenv("CF_ACCESS_CLIENT_ID", ""),
        "CF-Access-Client-Secret": os.getenv("CF_ACCESS_CLIENT_SECRET", ""),
    },
)

response = client.chat(
    model="qwen3-coder-next:latest", messages=[{"role": "user", "content": "Hello"}]
)

thinking = response["message"].get("thinking")
if thinking:
    print("=== Thinking ===")
    print(thinking)
    print("=== Response ===")

print(response["message"]["content"])
