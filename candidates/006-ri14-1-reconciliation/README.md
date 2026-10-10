# Person reconciliation on historical digests with the DH Infra LLM API

A small, complete example of using the DH Infra API from Python: link the persons
mentioned in a historical text to a catalogue of known persons, or decide that a person is
new. The texts are digests (*Regesten*) of King Maximilian I., 1493–1495, from the Regesta
Imperii; the catalogue and the answer key come from the printed index to that volume. Every
run is scored against that answer key, so you can see what the model gets right.

This is the task that prosopographical databases face every day: a new source arrives, and
each person in it either already has a record or needs one. The method comes from the
Managing Maximilian project, https://managing-maximilian.net (see *Credits*).

## What it does

The input is a digest and the person mentions in it (`mentions` in `test.jsonl`), as the
RI XIV,1 Register Dataset records them: the mention strings of the persons the printed index
attaches to the digest, located in the text and reviewed in the dataset's curation. The task
is reconciliation only: for each mention, decide which catalogue person it is, or that the
person is not in the catalogue.

For each mention:

1. **Candidates**: the catalogue is searched by name similarity (rapidfuzz, with the index's
   spelling variants, the mention forms seen in catalogue digests, and titles as name forms,
   weighted by rarity, so "Absperg" finds "Absberg", a surname counts more than "Hans", and
   "der Papst" finds the popes). No API call.
2. **Expansion, only when the search has not really found the person**: if the best
   candidate's name is less than 70 % similar to the mention, or there is no candidate, the
   mention is one a name index cannot resolve ("frz. Kg", "die spanischen Kge", "KMs Vater",
   a bare "Ulrich"). One API call then asks who is meant, as the index would write the name,
   from the digest first and general historical knowledge second, and the expanded names go
   through the same search. The model only proposes search strings; it cannot invent a
   catalogue person. The threshold comes from the reference run: below 70 the decisions
   were wrong more often than right, above it 94 % correct.
3. **Judge**: a mention without any candidate is "new" without a call. Otherwise one API call
   shows the digest, the mention, the expansion hint if there was one, and up to ten
   candidates with their index label, known spellings, places, dates and example digests,
   and asks for `match` (with the ids; a plural mention may cover several persons), `new` or
   `unsure`.

Every API response is cached on disk, so a repeated run makes no calls.

Persons the index did not list are not part of the task, and index links whose person does
not occur in the exported digest text (`absent_in_text` in `test.jsonl`, mostly co-addressees
named only in the archival note) are excluded. Extraction of mentions from raw text is a
separate task that this example does not cover.

## What you need

- **Session:** an interactive shell or a JupyterHub terminal. CPU only; the model runs on
  the cluster's gateway, this code only sends requests.
- **Resources:** no GPU; a few hundred MB of RAM; 3 MB of data; about 140 API calls for the
  pilot and about 2,100 for the full test set.
- **Software:** Python 3.10 or newer. The setup step installs `openai` and `rapidfuzz`. You
  need a DH Infra API key.

## Quickstart

```bash
pip install -r requirements.txt            # openai, rapidfuzz
cp .env.example .env                       # put your DH Infra API key in .env
python dhinfra.py                          # lists the chat models served right now
python reconcile.py --run pilot            # 50 digests, about 140 API calls, 2 minutes
python evaluate.py --run pilot             # scores runs/pilot/predictions.jsonl
```

`--run test` processes all 668 test digests (about 2,100 calls, 25 minutes with the default
16 workers). Every response is cached on disk and its token usage recorded; a repeated run
makes no calls, an interrupted run is resumed by repeating the command, and
`runs/<run>/run.json` sums what the run consumed. Errors follow the DH Infra docs: a rejected
key or a spent budget aborts the run, rate limits and busy models are retried.

## What's inside

| File | What it is |
|---|---|
| `reconcile.py` | the example: candidate search and judging per mention, results to `runs/<run>/predictions.jsonl` |
| `evaluate.py` | scores a run against the gold: precision, recall and F1 of both decisions, writes `runs/<run>/evaluation.md` |
| `dhinfra.py` | API client on the openai library: key from `.env`, thinking budget, retries, response cache with usage, `served_models()` |
| `prompts/judge.md`, `prompts/expand.md` | the two prompts, plain text |
| `data/catalogue.jsonl` | 2,046 known persons with profiles built only from the catalogue digests |
| `data/test.jsonl` | 668 digests, each with `mentions` (the input: the person mention strings in the text) and `gold` (the answer key: for each person its catalogue id, or "new") |
| `data/pilot_50.txt` | 50 test digests chosen so that rare and new persons are present |
| `data/LICENSE.txt` | the licence of the data, see below |

## Settings to change before you run

- `.env`: your DH Infra API key, one line, see `.env.example`. Nothing else points at the
  author's setup; all paths are relative to the scripts.
- `--model SLUG`: a served model. The default in `dhinfra.py` is the one served when this was
  written; `python dhinfra.py` lists what is served now.
- `--thinking on|off` and `--thinking-budget N`: reasoning on by default, bounded by a
  thinking token budget. `--workers N`: parallel requests, 16 by default; lower it if you
  see 429 responses. `--limit N`, `--only "<digest id>" ...`: run part of the set.

## What it produces

A run writes `runs/<run>/predictions.jsonl` (every decision with its candidates and the
model's reason), `evaluation.md` (the numbers below and a breakdown by how well the person
is covered in the catalogue) and `run.json` (settings and token usage).

Full test set, 668 digests, 2,059 decisions, deepseek-v4.1-flash, 16 workers. A **match** is
correct when the system returns the gold person's catalogue id, a **new** when the person is
not in the catalogue.

| thinking | match P / R / F1 | new P / R / F1 | calls | min | M tokens in / out |
|---|---|---|---|---|---|
| on, 4,096 budget | 98 / 95 / 96 | 79 / 95 / 86 | 2,116 | 26 | 4.2 / 1.02 |
| off | 97 / 94 / 96 | 81 / 95 / 88 | 2,111 | 18 | 4.1 / 0.14 |

Thinking changes the result little and the output tokens sevenfold.

## Data

The three data files are derived from the [RI XIV,1 Register Dataset](https://github.com/suzanaSagadin/ri14-1_register_dataset),
which in turn is built from the Regesta Imperii data published by the Akademie der
Wissenschaften und der Literatur Mainz in its RI Lab repository,
https://gitlab.rlp.net/adwmainz/regesta-imperii/lab/regesta-imperii-data. They are the
complete task, not a slice; nothing here needs to be regenerated. The dataset's README gives
the citations of the Regesta Imperii volume, its index and the RI Online data.

The digests and the index behind `data/` are Regesta Imperii material, published under
CC BY 4.0 by Regesta Imperii Online. The data files contain that published material and
what the dataset adds to it: entity identifiers, entity types, the catalogue profiles, the
split, and the mention strings, which were located automatically in the digest texts and
reviewed with Claude models (Fable 5.1 for the mention strings, Sonnet 5.5 and Opus 5.5 for
the types), as the dataset's README documents. Nothing comes from unpublished sources.

## Licence

Code: Apache-2.0, the repository default. Documentation and prompts (`README.md`,
`prompts/`): CC BY 4.0. Data (`data/`): CC BY 4.0, see `data/LICENSE.txt`; the third-party
material in it is Regesta Imperii Online data (Akademie der Wissenschaften und der Literatur
Mainz), CC BY 4.0, and the RI XIV,1 Register Dataset (Suzana Sagadin, 2026), CC BY 4.0, with
the attribution given in `data/LICENSE.txt`.

## Credits

This reconciliation method was developed as part of the Managing Maximilian (ManMax) project
at the Institute for Digital Humanities, University of Graz, supported by the Austrian Science
Fund (FWF) within the Special Research Programme SFB F92 Managing Maximilian
(DOI: 10.55776/F92). In the project, an LLM pipeline turns historical source texts into
records of a prosopographical database; its last step links every extracted person to the
database, by name candidates and a model judging each mention with the source text and the
candidates' profiles. This example is that step on public data.

Runs were made on the model gateway of Digital Humanities Infrastructure Austria (DHinfra.at),
https://api.dhinfra.uni-graz.at, with the served model deepseek-v4.1-flash. Tools:
[openai-python](https://github.com/openai/openai-python), [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz).
Data: [Regesta Imperii Online](https://www.regesta-imperii.de) and the
[RI XIV,1 Register Dataset](https://github.com/suzanaSagadin/ri14-1_register_dataset).

**How to cite.** This use case:

- Suzana Sagadin, Person reconciliation on historical digests with the DH Infra LLM API,
  2026, Apache-2.0 / CC BY 4.0, in: DH Infra use cases, https://github.com/dhinfra-at/use-cases.

For the data, cite the [RI XIV,1 Register Dataset](https://github.com/suzanaSagadin/ri14-1_register_dataset),
whose README gives the citations of the Regesta Imperii volume, its index and the RI Online data.
