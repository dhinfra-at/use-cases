"""Helpers for the two notebooks: prompt assembly, one model call, output checks."""

import json
import os
import re
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from lxml import etree
from openai import OpenAI

BASE_URL = os.environ.get("DHINFRA_BASE_URL", "https://api.dhinfra.uni-graz.at/v1")


def get_client():
    """OpenAI client for the DHinfra gateway. The token is read from DHINFRA_KEY (env or .env)."""
    # override=True: a token in .env wins over one already set in the environment.
    load_dotenv(find_dotenv(usecwd=True), override=True)
    key = os.environ.get("DHINFRA_KEY")
    if not key:
        raise SystemExit(
            "DHINFRA_KEY is not set. Put your project token in a .env file, "
            'as DHINFRA_KEY="...", or export it in your session.'
        )
    return OpenAI(api_key=key, base_url=BASE_URL)


def build_system_prompt(prompt_dir):
    """prompt.txt + encoding_rules.txt + few_shot_examples.txt, joined by blank lines."""
    parts = []
    for name in ("prompt.txt", "encoding_rules.txt", "few_shot_examples.txt"):
        parts.append((Path(prompt_dir) / name).read_text(encoding="utf-8").strip())
    return "\n\n".join(parts)


def build_user_message(prompt_dir, letter_text):
    """The short instruction from user_message.txt, then the letter."""
    prefix = (Path(prompt_dir) / "user_message.txt").read_text(encoding="utf-8").strip()
    return f"{prefix}\n\n{letter_text.strip()}"


def clean_output(raw):
    """Remove <think> blocks and a surrounding ```xml fence, if the model added them."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    fence = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n```\s*$", text, flags=re.DOTALL)
    return fence.group(1).strip() if fence else text


def ask_model(client, model, system_prompt, user_message, thinking=False,
              temperature=0.3, max_tokens=8000):
    """One request to the model. Returns (answer_text, usage)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        # vLLM chat-template switch for the thinking mode (models without one ignore it).
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    return clean_output(response.choices[0].message.content or ""), response.usage


def _words(text):
    return re.findall(r"\w+", text, flags=re.UNICODE)


def check_output(xml_text, source_text):
    """Cheap checks on one result: well-formed, text kept, which letter elements are used."""
    result = {"well_formed": False, "text_kept_%": None, "elements": ""}
    try:
        root = etree.fromstring(xml_text.encode("utf-8"))
    except etree.XMLSyntaxError as err:
        result["error"] = str(err)
        return result
    result["well_formed"] = True
    # Is every word of the source still in the output, in the same order? (Counts the share of
    # source words that the output keeps; the model must not shorten or paraphrase the letter.)
    out_words = _words(" ".join(root.itertext()))
    src_words = _words(source_text)
    kept = 0
    pos = 0
    for word in src_words:
        try:
            pos = out_words.index(word, pos) + 1
            kept += 1
        except ValueError:
            continue
    result["text_kept_%"] = round(100 * kept / max(len(src_words), 1), 1)
    tags = ["opener", "dateline", "salute", "p", "closer", "signed", "postscript", "address"]
    counts = {t: len(root.findall(f".//{t}")) for t in tags}
    result["elements"] = " ".join(f"{t}:{n}" for t, n in counts.items() if n)
    return result


# ---------------------------------------------------------------------------
# Editorial interventions: a bracketed sequence in, its type and TEI encoding out
# ---------------------------------------------------------------------------

INTERVENTION_TYPES = {"Abbreviation", "Correction", "Supplement", "Omission",
                      "Uncertainty", "Redundant Text"}
ALLOWED_TAGS = {"choice", "abbr", "expan", "sic", "corr", "supplied", "gap", "unclear", "surplus"}


def build_items_message(prompt_dir, context, items):
    """Instruction, the context sentence, and the bracketed sequences (a list if there are several)."""
    prefix = (Path(prompt_dir) / "user_message.txt").read_text(encoding="utf-8").strip()
    listed = items[0] if len(items) == 1 else "\n".join(f"- {item}" for item in items)
    return f'{prefix}\n\nContext: "{context}"\n\nBracketed Sequences:\n{listed}'


def parse_json_answer(raw):
    """The answer should be a JSON object {sequence: {type, tei}}. Returns a dict or None."""
    text = clean_output(raw)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def check_interventions(answer, items):
    """One row per bracketed sequence: answered? type named in the list of six? TEI well-formed and allowed?

    These are format checks only. They say nothing about whether the type or the TEI is correct."""
    rows = []
    for item in items:
        entry = (answer or {}).get(item)
        row = {"sequence": item, "answered": isinstance(entry, dict),
               "type": None, "type_in_list": False, "tei_ok": False, "tei": None}
        if row["answered"]:
            row["type"] = entry.get("type")
            row["tei"] = entry.get("tei")
            row["type_in_list"] = all(t.strip() in INTERVENTION_TYPES
                                    for t in (row["type"] or "").split(";"))
            try:
                fragment = etree.fromstring(f"<x>{row['tei']}</x>")
                tags = {el.tag for el in fragment.iter() if el is not fragment}
                has_attributes = any(el.attrib for el in fragment.iter())
                row["tei_ok"] = tags <= ALLOWED_TAGS and not has_attributes
            except (etree.XMLSyntaxError, TypeError):
                pass
        rows.append(row)
    return rows
