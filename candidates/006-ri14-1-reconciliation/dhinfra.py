"""Minimal client for the DH Infra LLM API (OpenAI-compatible chat completions).

Reads DHINFRA_API_KEY from the environment or a .env file next to this module.
Uses the openai client as the DH Infra docs show. Every response is cached on
disk under runs/<run>/cache/, keyed by the full request, so a repeated run
makes no API calls; token usage is recorded per response and summed per run.

Reasoning: on by default with a thinking budget (DH Infra's
`thinking_token_budget`), so the model thinks a bounded amount before the
answer and `max_tokens` is budget + answer room. `enable_thinking=False`
turns it off (`chat_template_kwargs`).

Errors, as the DH Infra docs define them: 401 (bad or revoked key) and 402
(budget spent) abort the run; 429 waits as long as Retry-After says; 503 and
other 5xx are retried shortly; 404 stops with the list of served models;
400 is our own mistake and is raised.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from openai import (OpenAI, APIStatusError, APIConnectionError, APITimeoutError, AuthenticationError,
                    BadRequestError, NotFoundError, RateLimitError)

BASE_URL = "https://api.dhinfra.uni-graz.at/v1"
MODEL = "deepseek-v4.1-flash"   # served models change; check with `python dhinfra.py` or --model
RETRY_WAITS = (5, 15, 30, 60, 120)
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens")


def _load_dotenv():
    p = Path(__file__).resolve().parent / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class DHInfra:
    def __init__(self, cache_dir, model=MODEL, base_url=BASE_URL, temperature=0.0,
                 enable_thinking=True, thinking_budget=4096, answer_tokens=1024, timeout=600):
        _load_dotenv()
        key = os.environ.get("DHINFRA_API_KEY")
        if not key:
            raise SystemExit("DHINFRA_API_KEY is not set (see .env.example)")
        # the client retries 429/5xx by itself (honouring Retry-After); the loop below adds a longer tail
        self.client = OpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=2)
        self.model, self.temperature = model, temperature
        self.enable_thinking, self.thinking_budget, self.answer_tokens = enable_thinking, thinking_budget, answer_tokens
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.calls = self.cache_hits = 0
        self.usage = dict.fromkeys(USAGE_KEYS, 0)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ request
    def _payload(self, system, user):
        p = {"model": self.model, "temperature": self.temperature,
             "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.enable_thinking:
            p["max_tokens"] = self.thinking_budget + self.answer_tokens
            p["extra_body"] = {"thinking_token_budget": self.thinking_budget}
        else:
            p["max_tokens"] = self.answer_tokens
            p["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return p

    def chat(self, system, user):
        """-> (content, reasoning). Cached by model + settings + messages."""
        payload = self._payload(system, user)
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        hit = self.cache / f"{key}.json"
        if hit.exists():
            d = json.loads(hit.read_text(encoding="utf-8"))
            with self._lock:
                self.cache_hits += 1
            return d["content"], d.get("reasoning")
        content, reasoning, usage, finish = self._request(payload)
        if not content.strip() and finish == "length":
            # the answer did not fit: once more with twice the room, same thinking budget
            content, reasoning, usage, finish = self._request({**payload, "max_tokens": payload["max_tokens"] * 2})
        with self._lock:
            self.calls += 1
            for k in USAGE_KEYS:
                self.usage[k] += usage.get(k, 0)
        if content.strip():          # an empty answer is not cached, so the next run retries it
            hit.write_text(json.dumps({"request": payload, "content": content, "reasoning": reasoning,
                                       "usage": usage, "finish_reason": finish}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        return content, reasoning

    def _request(self, payload):
        for attempt in range(len(RETRY_WAITS) + 1):
            try:
                r = self.client.chat.completions.create(**payload)
                m = r.choices[0].message
                reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
                u = r.usage
                usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                         "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0,
                         "cached_tokens": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0}
                return m.content or "", reasoning, usage, r.choices[0].finish_reason
            except AuthenticationError as e:                      # 401
                raise SystemExit(f"API key rejected (401): {e.message}. Put a valid DHINFRA_API_KEY into .env.")
            except NotFoundError:                                 # 404
                raise SystemExit(f"model {self.model!r} is not served; available: {', '.join(self.served_models())}")
            except BadRequestError as e:                          # 400: our request is wrong
                raise
            except RateLimitError as e:                           # 429 after the client's own retries
                wait = _retry_after(e) or RETRY_WAITS[min(attempt, len(RETRY_WAITS) - 1)]
            except APIStatusError as e:
                if e.status_code == 402:                          # budget spent: retrying will not help
                    raise SystemExit(f"token budget spent (402): {e.message}")
                if e.status_code < 500:
                    raise
                wait = RETRY_WAITS[min(attempt, len(RETRY_WAITS) - 1)]   # 503 busy, 5xx: retry shortly
            except (APIConnectionError, APITimeoutError):
                wait = RETRY_WAITS[min(attempt, len(RETRY_WAITS) - 1)]
            if attempt == len(RETRY_WAITS):
                raise RuntimeError("request failed after retries")
            time.sleep(wait)
        raise RuntimeError("unreachable")

    def served_models(self):
        return [m.id for m in self.client.models.list().data
                if not any(k in m.id for k in ("embed", "minilm", "bge", "jina", "rerank"))]

    def settings(self):
        return {"model": self.model, "temperature": self.temperature, "thinking": self.enable_thinking,
                "thinking_budget": self.thinking_budget if self.enable_thinking else 0,
                "answer_tokens": self.answer_tokens}


def _retry_after(e):
    try:
        return float(e.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def parse_json(text):
    """Extract the first JSON object or array from a model answer."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    starts = [(text.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if text.find(o) >= 0]
    for i, opener, closer in sorted(starts):
        j = text.rfind(closer)
        if j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON in model answer: " + text[:200])


if __name__ == "__main__":
    import tempfile
    print("served chat models:", ", ".join(DHInfra(tempfile.mkdtemp()).served_models()))
