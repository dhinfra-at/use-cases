# LLM Processing of Historical Accounting Book Data — JSONtoRDF

Encoding transcribed medieval accounting book entries into RDF/XML with an LLM, and
measuring the result against a manually encoded ground truth.

This repository is a **reduced, standalone extract** of the LLM pipeline built for the
project *Aldersbach Digital* (University of Graz). It contains one step of that pipeline —
**JSONtoRDF** — one provider — the **DH-Infra cluster** of the University of Graz — and
**ground-truth data only**. Everything it reads and writes stays inside this folder.

link to author repo: https://github.com/MaxVogeltanz/LLMProcessing_HistoricalAccountingBookData

Author: Maximilian Vogeltanz, University of Graz, 2026

---

## What it does

A JSON array of transcribed account-book entries goes in; RDF/XML following the
Aldersbach data model (`bk:` bookkeeping / `arec:` ontology) comes out.

```
data/standalone_entries_L343GT.json
        │
        │  LLM_Processor.py                  one entry per prompt, in parallel
        ▼
output/JSONtoRDF/raw/L343GT_<model>.xml        raw model output
        │
        │  output/JSONtoRDF/postprocess_RDF_output.py
        ▼
output/JSONtoRDF/postprocessed/L343GT_<model>.xml
        │
        │  output/JSONtoRDF/compare_RDF.py   against data/GroundTruth/
        ▼
output/JSONtoRDF/evaluation/L343_<model>.html
        │
        │  output/JSONtoRDF/evaluation/generate_summary.py
        ▼
output/JSONtoRDF/evaluation/evaluation_jsontordf_total.html
```

### The division of labour between model and code

The model is good at *selecting* and bad at *computing*, so the pipeline splits the two:

* **Dates.** The model does not compute a calendar date. It copies the date phrase out of
  the entry into `<bk:when resolve-from="…"/>`; `latin_dates.py` then resolves it
  deterministically (Julian calendar, use of the Diocese of Passau). Measured on L343, the
  model's own arithmetic agreed with the correct date in roughly a third of cases — the
  recurring error is `ante`/`post` landing a week off the feast's weekday.
* **Numerals.** `latin_amounts.py` reproduces the model's known misreading of the halving
  stroke `j̸` and corrects a quantity only where it matches exactly, so selection stays
  with the model.

Both are pure functions of their input: the same phrase always yields the same date, the
same numeral always the same value, and neither consults a model.

---

## Setup

```
pip install openai python-dotenv PyYAML lxml rdflib
```

Put your project bearer token into `.env` in the repository root, as `DHINFRA_KEY`, from
[console.dhinfra.uni-graz.at](https://console.dhinfra.uni-graz.at):

```
DHINFRA_KEY="…"
```

The cluster exposes an OpenAI-compatible vLLM API at
`https://api.dhinfra.uni-graz.at/v1` (override with `DHINFRA_BASE_URL`).

To check that the key works — and as the quickest way to try a model or a prompt wording —
run the single-call example, which has no config file and no input or output files: set
`MODEL`, `SYSTEM_PROMPT` and `USER_MESSAGE` at the top of it and the answer is printed to
the console.

```
python LLM_Processor_example.py
```

---

## Running it

**1 — Encode.** Set the accounting book, the model and the paths in `config.yaml`, then:

```
python LLM_Processor.py
```

Writes `output/JSONtoRDF/raw/L343GT_<model>.xml` and a run log (model, prompt hash, token
counts, throughput) to `output/JSONtoRDF/logs/`. A batch that fails all retries is left in
the output as `/* ERROR: batch N failed */` and reported at the end — search for that
string before using the file. Each request carries one entry, so a "batch" in `config.yaml`
and in the console output is one entry.

**2 — Postprocess.** Adapt the three paths at the top of the script (marked
`adapt as needed`), then:

```
python output/JSONtoRDF/postprocess_RDF_output.py
```

Comments out non-XML chatter, wraps the blocks in an RDF header, adds
`gams:isMemberOfCollection` / `gams:isPartOf`, splits multi-node properties, resets
back-reference words (`idem`, `eidem`, …) used as agent ids to `#anonymous`, resolves the
date phrases, validates the RDF against the input JSON (ids, `bk:entry`, `arec:hasRubric`,
year), corrects `j̸` misreadings and checks well-formedness. Differences against the JSON
can be corrected interactively when the script runs in a terminal.

**3 — Evaluate.** Adapt the paths and `rb_number` at the top, then:

```
python output/JSONtoRDF/compare_RDF.py
```

Produces a per-entry side-by-side HTML diff against the ground truth plus three F1 scores
— structural, content fidelity and exact triple string — each as macro- and micro-average.

**4 — Summarise.** 

```
python output/JSONtoRDF/evaluation/generate_summary.py
```

Consolidates every `L<book>_<model>.html` in `evaluation/` into
`evaluation_jsontordf_total.html`. The cross-dataset average is weighted by the number of
entries per book (counted at run time from the ground truth), so every entry counts the
same regardless of which volume it comes from.

### The results already in the repository

`output/JSONtoRDF/evaluation/` is not empty: it ships 20 reports from earlier runs of this
pipeline — the four books against five configurations, namely `glm5.1-fp8`, `kimi-k2.5`
and `qwen3.5-397b`, the latter two each also with thinking enabled — plus the consolidated
`evaluation_jsontordf_total.html` built from them.

They are demonstration material, from earlier runs of the pipeline: they show what the
evaluation produces and roughly how the models compare, not a benchmark tied to the exact
prompt revision in `data/prompts/` as it now stands.

They are there so that the evaluation can be looked at without an API key and without
re-running anything: open the consolidated file to compare the models, or an individual
`L<book>_<model>.html` to see the per-entry diffs against the ground truth. Running
`generate_summary.py` rebuilds the consolidated file from whatever is in the folder, so
your own runs are added to the comparison rather than replacing it — as long as the report
is named `L<book>_<model>.html`.

---

## Layout

```
config.yaml                          model, paths, run parameters
LLM_Processor.py                     the encoding run
LLM_Processor_example.py             one call, prompt in a variable, no config
providers/                           DH-Infra client (base.py, dhinfra_client.py)
data/
  standalone_entries_L343GT…L346GT.json   the entries to encode (ground truth)
  GroundTruth/                       manually encoded RDF/XML, reduced
  prompts/systemprompt_JSONtoRDF.txt the system prompt (data model + examples)
output/JSONtoRDF/
  postprocess_RDF_output.py          raw output -> valid, enriched RDF
  compare_RDF.py                     postprocessed output vs. ground truth
  latin_dates.py                     deterministic date resolution
  latin_amounts.py                   deterministic numeral/amount reading
  raw/ postprocessed/ logs/          run artefacts, created on the first run
  evaluation/                        per-run HTML reports (20 included) + generate_summary.py
```

---

## The data

Four Latin account books of the Cistercian monastery of Aldersbach, kept in the Bavarian
Main State Archive (D-MBayHStA, KAAA):

| id | archival unit | years | entries |
|------|---------------|-------|---------|
| L343 | KAAA 343 | 1455–57 | 205 |
| L344 | KAAA 344 | 1458–62 | 59 |
| L345 | KAAA 345 | 1463–64 | 73 |
| L346 | KAAA 346 | 1466–72 | 38 |

Each entry carries `id`, `rubric`, `year`, `type` and the transcription in `entry`. The
ground-truth RDF in `data/GroundTruth/` is a reduced version of the project's manually
encoded material and carries no `arec:recordedBy` — the pointer into the TEI source exists
only when whole books are processed, not in the ground truth, so the prompt does not ask
for it.

---

## Relation to the full pipeline

The complete *Aldersbach Digital* pipeline has further steps (entry splitting, entry
classification, RDF review passes) and further providers, and it processes the full books
rather than the ground truth. None of that is part of this repository, and this repository
never reads from or writes to it.

---

## How to cite

If you use this code or the ground-truth encodings, please cite the repository:

> Vogeltanz, Maximilian (2026): *LLM Processing of Historical Accounting Book Data — JSONtoRDF.*
> University of Graz.
> https://github.com/MaxVogeltanz/LLMProcessing_HistoricalAccountingBookData

---

## License

Two licenses, because the repository holds two different kinds of thing:

* **Code** — everything under `providers/`, `output/JSONtoRDF/*.py`, `LLM_Processor.py`
  and `config.yaml`: [Apache License 2.0](LICENSE).
* **Data** — the entries in `data/`, the ground-truth RDF in `data/GroundTruth/`, the
  system prompt and the evaluation reports in `output/JSONtoRDF/evaluation/`:
  [CC BY 4.0](LICENSE-DATA). The same notice sits in [`data/LICENSE`](data/LICENSE), so
  the terms travel with the data if the folder is passed on by itself.

Copyright 2026 Maximilian Vogeltanz, University of Graz. The underlying manuscripts are held by
the Bayerisches Hauptstaatsarchiv, Klosterarchiv Aldersbach (KAAA 343–346); the license
covers the transcriptions and encodings, not the originals.
