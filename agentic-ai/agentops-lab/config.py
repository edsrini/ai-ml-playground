import os

from dotenv import load_dotenv

load_dotenv()

# Any model already pulled via `ollama pull <name>` works here.
# Available locally at time of setup: qwen3:4b, deepseek-r1:latest, llama3.2:1b
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
