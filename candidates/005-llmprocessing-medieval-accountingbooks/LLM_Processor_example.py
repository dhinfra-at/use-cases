# Author: Maximilian Vogeltanz, University of Graz, 2026

# Minimal, self-contained example of a single call to the DH-Infra cluster. No config file,
# no input or output files: set the variables below and the answer is printed to the console.
# Use it to try a model out, to test an API key, or to iterate on prompt wording. For the
# real encoding run use LLM_Processor.py, which reads config.yaml.

from pathlib import Path
from dotenv import load_dotenv
from providers import make_client

# ─────────────────────────────── edit this part ───────────────────────────────

MODEL = "qwen3.5-397b"   # see your DH-Infra dashboard for the models available to you

SYSTEM_PROMPT = """\
You are an helpful and friendly AI Agent
"""

USER_MESSAGE = """\
Hello, there.
"""

MAX_TOKENS = 4096
TEMPERATURE = 0
THINKING = False   # the client maps this to the right parameter per model family

# ──────────────────────────── nothing to edit below ───────────────────────────


def main():
    # This repository's own .env, not one that happens to sit in a parent folder.
    load_dotenv(Path(__file__).resolve().parent / ".env")

    client = make_client("dhinfra")

    print(f"Model:   {MODEL}")
    print(f"Thinking: {THINKING}")
    print("Waiting for the model …\n")

    result = client.generate(
        system=SYSTEM_PROMPT,
        user=USER_MESSAGE,
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        thinking={"enabled": THINKING},
    )

    print("─" * 78)
    print(result.text)
    print("─" * 78)
    print(f"Tokens — input: {result.usage.input_tokens}, "
          f"output: {result.usage.output_tokens}")
    

if __name__ == "__main__":
    main()
