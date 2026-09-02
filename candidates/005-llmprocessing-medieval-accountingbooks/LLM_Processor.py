# Author: Maximilian Vogeltanz, University of Graz, 2026

# Script to encode json files containing transcribed accounting book entries into RDF/XML
# through LLM processing for project "Aldersbach digital".
# The system prompt (stored in data/prompts) iterates over each defined amount of json objects
# (batches); input is JSON, output is RDF/XML.
#
# This standalone repository covers the JSONtoRDF step only, and runs against the DH-Infra
# cluster of the University of Graz (see providers/dhinfra_client.py).
#
# PIPELINE ORDER:
#   LLM_Processor.py  ->  output/JSONtoRDF/postprocess_RDF_output.py  ->  output/JSONtoRDF/compare_RDF.py


import time
import json
from pathlib import Path
from datetime import datetime, timezone
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import yaml
from providers import make_client


BASE_DIR = Path(__file__).resolve().parent

# The only provider in this repository. Not a config key: there is nothing to choose
# between, and the run log records it as provenance either way.
PROVIDER = "dhinfra"

# One json object per request. For RDF encoding this is the only sensible value: several
# entries in one prompt cost context isolation and risk a max_tokens cutoff in the middle
# of a document, with no gain — the requests run in parallel anyway.
OBJECTS_PER_BATCH = 1

# ---------- helpers ----------
def load_text_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"⚠️ Warning: file not found: {path}. Proceeding without it.")
        return ""

def load_json_file(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def chunk_objects(objects, size: int):
    for i in range(0, len(objects), size):
        yield objects[i : i + size]

def format_path(template: str, *, provider: str, model: str) -> str:
    safe_model = model.replace("/", "_").replace(":", "_")
    return (
        template
        .replace("{PROVIDER}", provider.lower())
        .replace("{MODEL}", "_" + safe_model)
    )

# ---------- main ----------
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def write_run_log(log_path: Path, log_data: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    start_time = time.time()

    # Explicitly this repository's own .env. A bare load_dotenv() searches upward from the
    # current working directory and would pick up a .env of a PARENT folder when the script
    # is started from somewhere else — a key from an unrelated project, silently.
    load_dotenv(BASE_DIR / ".env")

    CONFIG_PATH = BASE_DIR / "config.yaml"

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    provider = PROVIDER
    accountingbook = cfg["accountingbook"]
    paths = cfg["paths"]
    run = cfg["run"]
    gen = cfg["generation"]

    model_name = gen["model"]
    max_tokens = int(gen.get("max_tokens", 4096))
    temperature = float(gen.get("temperature", 0))

    input_file_path = (BASE_DIR / paths["input"]).resolve() # input file path as defined in config.yaml
    system_prompt_path = (BASE_DIR / paths["system_prompt"]).resolve() # path to system prompt as defined in config.yaml
    output_file_path = (BASE_DIR / format_path(paths["output"], provider=provider, model=model_name)).resolve() # path to output file as defined in config.yaml

    system_prompt_text = load_text_file(system_prompt_path)
    system_prompt = f"{system_prompt_text}\n"

    objects_per_batch = OBJECTS_PER_BATCH
    batch_range = str(run.get("batch_range", "")).strip()
    max_retries = int(run.get("max_retries", 3))
    backoff_base = float(run.get("retry_backoff_base_seconds", 2))
    fail_on_empty = bool(run.get("fail_on_empty_response", False))
    max_workers = int(run.get("max_workers", 1))

    thinking_config = gen.get("thinking", None)
    thinking_enabled = thinking_config.get("enabled", False) if isinstance(thinking_config, dict) else False

    start_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time:     {start_iso}")
    print(f"Provider:       {provider}")
    print(f"Model:          {model_name}")
    print(f"Input:          {input_file_path}")
    print(f"Output:         {output_file_path}")
    print(f"System prompt:  {system_prompt_path}")
    print(f"Thinking:       {thinking_enabled}")

    client = make_client(provider)

    json_objects = load_json_file(input_file_path)
    print(f"Loaded {len(json_objects)} JSON objects")

    total_batches = (len(json_objects) + objects_per_batch - 1) // objects_per_batch

    if batch_range:
        start, end = (int(x) for x in batch_range.split("-"))
        end = min(end, total_batches)
        json_objects = json_objects[start * objects_per_batch : end * objects_per_batch]
        print(f"⚙️ Limited to batches {start}-{end} of {total_batches} ({len(json_objects)} objects)")
        total_batches = end - start

    all_outputs = []
    total_input_tokens = 0
    total_output_tokens = 0

    # --- Synchronous mode (sequential or parallel). The DH-Infra cluster exposes no
    #     batch API, so every request is a normal chat completion. ---
    batches = list(enumerate(chunk_objects(json_objects, objects_per_batch), start=1))
    failed_batches = []  # collect (batch_index, last_error) for summary

    def process_batch_tracked(batch_index, batch_objects):
        batch_text = json.dumps(batch_objects, ensure_ascii=False, indent=2)
        last_error = None
        for attempt in range(max_retries):
            try:
                print(f"🟡 Batch {batch_index}/{total_batches} – Attempt {attempt + 1}")
                result = client.generate(
                    system=system_prompt,
                    user=batch_text,
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking=thinking_config
                )
                if fail_on_empty and not result.text.strip():
                    raise ValueError("Empty response from model")
                print(f"✅ Batch {batch_index}/{total_batches} – done")
                return batch_index, result.text, result.usage.input_tokens, result.usage.output_tokens, None
            except Exception as e:
                last_error = e
                print(f"⚠️ Error in batch {batch_index}, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff_base ** attempt)
        return batch_index, f"\n/* ERROR: batch {batch_index} failed */\n{batch_text}", 0, 0, last_error

    if max_workers > 1:
        print(f"⚡ Parallel mode: {max_workers} workers")
        results_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_batch_tracked, idx, objs): idx for idx, objs in batches}
            for future in as_completed(futures):
                idx, text, in_tok, out_tok, err = future.result()
                results_map[idx] = (text, in_tok, out_tok, err)
        for idx in sorted(results_map):
            text, in_tok, out_tok, err = results_map[idx]
            all_outputs.append(text)
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            if err is not None:
                failed_batches.append((idx, err))
    else:
        for idx, objs in batches:
            _, text, in_tok, out_tok, err = process_batch_tracked(idx, objs)
            all_outputs.append(text)
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            if err is not None:
                failed_batches.append((idx, err))

    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file_path).write_text("\n\n".join(all_outputs), encoding="utf-8")

    total_tokens = total_input_tokens + total_output_tokens

    print("\n--- Token Usage Summary ---")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total tokens used: {total_tokens}")

    end_time = time.time()

    # Reported LAST, not before the token block: a failure that scrolls past and is then
    # followed by a green checkmark reads as a clean run. These failures ARE marked in the
    # artifact (the batch is replaced by a "/* ERROR: batch N failed */" comment), so the
    # point here is to make sure the operator looks for them.
    if failed_batches:
        print(f"\n{'!'*58}")
        print(f"❌ {len(failed_batches)} of {total_batches} batch(es) failed after all "
              f"{max_retries} attempts:")
        for batch_idx, error in failed_batches:
            print(f"   Batch {batch_idx}/{total_batches}: {error}")
        print(f"   Each stands in the output as '/* ERROR: batch N failed */' — search for "
              f"that string before using the file.")
        print(f"{'!'*58}")
        print(f"\n⚠️ Processing finished WITH {len(failed_batches)} FAILED BATCH(ES).")
    else:
        print("\n✅ Processing complete.")
    print(f"📝 Output written to: {output_file_path}")

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    avg_throughput = total_tokens / execution_time if execution_time > 0 else 0.0
    print(f"Average throughput: {avg_throughput:.2f} tokens/s")


    #Log File
    prompt_text_full = system_prompt_text  # already loaded
    prompt_hash = sha256_text(prompt_text_full)

    log_data = {
        "date_of_creation": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model_name,
        "input_file": str(input_file_path),
        "output_file": str(output_file_path),
        "system_prompt_file": str(system_prompt_path),
        "system_prompt_sha256": prompt_hash,
        "system_prompt_text": prompt_text_full,
        "run_params": {
            "objects_per_batch": objects_per_batch,
            "max_workers": max_workers,
            "max_retries": max_retries,
            "retry_backoff_base_seconds": backoff_base,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "results": {
            "num_input_objects": len(json_objects),
            "total_batches": total_batches,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "average_tokens_per_entry": (total_input_tokens + total_output_tokens)/len(json_objects),
            "execution_time (seconds)": execution_time,
            "average_throughput (tokens/s)": avg_throughput,
        },
    }

    # Put the log into a "logs" folder inside the current step's output folder
    # (here: output/JSONtoRDF/logs). The output path looks like
    # output/<step>/raw[/...]/file.ext, and the number of subfolders under <step> may
    # vary, so locate <step> robustly as the folder whose direct parent is "output"
    # rather than counting fixed levels.
    safe_model = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    step_dir = output_file_path.parent
    while step_dir.parent.name != "output" and step_dir.parent != step_dir:
        step_dir = step_dir.parent
    log_dir = step_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"log_{accountingbook}_{safe_model}_{timestamp}.json"

    write_run_log(log_path, log_data)
    print(f"🧾 Run log written to: {log_path}")


if __name__ == "__main__":
    main()
