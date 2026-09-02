# Author: Maximilian Vogeltanz, University of Graz, 2026

# Script for logic specific to models hosted on the DH-Infra cluster (University of Graz).
# vLLM exposes an OpenAI-compatible API at https://api.dhinfra.uni-graz.at/v1
# Authenticate with the project bearer token from console.dhinfra.uni-graz.at (DHINFRA_KEY in .env).
# Gets called in __init__.py in the same folder

import os
from openai import OpenAI
from .base import GenResult, Usage


class DhinfraClient:
    def __init__(self, base_url: str | None = None):
        base_url = base_url or os.getenv("DHINFRA_BASE_URL", "https://api.dhinfra.uni-graz.at/v1")
        api_key = os.getenv("DHINFRA_KEY", "dhinfra")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=1800)  # 30 min — large models can be slow

    def generate(self, *, system: str, user: str, model: str, max_tokens: int, temperature: float, thinking=None):
        thinking_enabled = bool(thinking and thinking.get("enabled"))
        # Thinking mode is model-specific:
        # - Kimi K2.5 uses the `thinking_config` parameter (type: "enabled"/"disabled")
        # - Qwen3 and others use `chat_template_kwargs` with `enable_thinking` (bool)
        # Both are controlled via `generation.thinking.enabled` in config.yaml
        if "kimi" in model.lower():
            extra_body = {"thinking_config": {"type": "enabled" if thinking_enabled else "disabled"}}
        else:
            extra_body = {"chat_template_kwargs": {"enable_thinking": thinking_enabled}}

        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
        choice = resp.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            print("⚠️ WARNING: finish_reason=length — output was cut off at max_tokens. Consider increasing max_tokens.")
        text = (choice.message.content or "").strip()
        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0),
            output_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
        return GenResult(text=text, usage=usage)
