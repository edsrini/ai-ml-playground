"""One-command setup for the class:  python bootstrap.py

Creates .venv, installs pinned dependencies, creates .env, checks Ollama and the model, and runs a
smoke test. Safe to re-run: every step skips work that is already done.
"""
import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
DEFAULT_MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434"


def step(message: str) -> None:
    print(f"\n==> {message}")


def fail(message: str) -> None:
    print(f"\nERROR: {message}")
    sys.exit(1)


def check_python() -> None:
    step("Checking Python version")
    if sys.version_info < (3, 10):
        fail(f"Python 3.10+ is required, found {sys.version.split()[0]}.")
    print(f"Python {sys.version.split()[0]} OK")


def create_venv() -> None:
    step("Creating virtual environment (.venv)")
    if VENV_PYTHON.exists():
        print("already exists, skipping")
        return
    venv.create(VENV_DIR, with_pip=True)
    print("created")


def install_dependencies() -> None:
    step("Installing dependencies from requirements.txt")
    result = subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    if result.returncode != 0:
        fail("pip install failed; see the output above.")


ENV_FILE = ROOT / ".env"


def set_env_model(model: str) -> None:
    """Write OLLAMA_MODEL=<model> into .env, replacing the existing line if there is one."""
    lines = ENV_FILE.read_text().splitlines()
    updated = [f"OLLAMA_MODEL={model}" if line.startswith("OLLAMA_MODEL=") else line for line in lines]
    if not any(line.startswith("OLLAMA_MODEL=") for line in lines):
        updated.append(f"OLLAMA_MODEL={model}")
    ENV_FILE.write_text("\n".join(updated) + "\n")


def model_from_env_file() -> str | None:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("OLLAMA_MODEL="):
                return line.split("=", 1)[1].strip() or None
    return None


def create_env_file(requested_model: str | None) -> None:
    step("Creating .env")
    if ENV_FILE.exists():
        print(".env already exists, leaving the rest of it alone")
    else:
        shutil.copy(ROOT / ".env.example", ENV_FILE)
        print("copied .env.example to .env")
    if requested_model:  # --model given: make the app use the same model we are about to download
        set_env_model(requested_model)
        print(f"set OLLAMA_MODEL={requested_model} in .env")


def ollama_models() -> list[str] | None:
    """Names of models the running Ollama server has, or None if the server isn't reachable."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as response:
            return [m["name"] for m in json.load(response)["models"]]
    except (urllib.error.URLError, TimeoutError):
        return None


def check_ollama(model: str, skip_model: bool) -> bool:
    step(f"Checking Ollama and the model ({model})")
    if not shutil.which("ollama"):
        print("Ollama is not installed. Install it from https://ollama.com/download, then re-run this script.")
        return False
    models = ollama_models()
    if models is None:
        print("Ollama is installed but not running. Start the Ollama app (or run `ollama serve`), then re-run.")
        return False
    if model in models:
        print(f"model {model} is already downloaded")
        return True
    if skip_model:
        print(f"model {model} is not downloaded (skipped; run `ollama pull {model}` yourself)")
        return False
    print(f"downloading {model} (a few GB, this can take a while)...")
    if subprocess.run(["ollama", "pull", model]).returncode != 0:
        print(f"`ollama pull {model}` failed.")
        return False
    return True


def smoke_test() -> bool:
    step("Running smoke test (imports and library tools, no model call)")
    code = (
        "from library import search_books; "
        "from langchain.agents import create_agent; "
        "from langchain_ollama import ChatOllama; "
        "out = search_books.invoke({'query': 'dune'}); "
        "assert 'Dune' in out, out; print(out)"
    )
    result = subprocess.run([str(VENV_PYTHON), "-c", code], cwd=ROOT)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the AI Library Assistant for class.")
    parser.add_argument("--model", help=f"Ollama model to use and save to .env (default: .env value, else {DEFAULT_MODEL})")
    parser.add_argument("--skip-model", action="store_true", help="don't download the model automatically")
    args = parser.parse_args()

    check_python()
    create_venv()
    install_dependencies()
    create_env_file(args.model)
    model = args.model or model_from_env_file() or DEFAULT_MODEL  # the model the app will actually use
    ollama_ready = check_ollama(model, args.skip_model)
    smoke_ok = smoke_test()

    activate = r".venv\Scripts\activate" if sys.platform == "win32" else "source .venv/bin/activate"
    print("\n" + "=" * 60)
    print(f"Dependencies installed: yes    Smoke test: {'passed' if smoke_ok else 'FAILED'}")
    print(f"Ollama + model ready:   {'yes' if ollama_ready else 'NO (see the messages above)'}")
    print(f"\nNext:  {activate}   then   python library_agent.py")
    sys.exit(0 if smoke_ok and ollama_ready else 1)


if __name__ == "__main__":
    main()
