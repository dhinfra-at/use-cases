# Encoding letters with an LLM: letter structure and editorial interventions

Two small TEI encoding tasks on transcribed 19th-century letters (German, French, English,
Italian) from the Hammer-Purgstall correspondence (1794-1856), solved with an LLM through the
cluster's gateway. Each has its own notebook:

1. **Letter structure.** Plain-text letters go in, TEI XML for the structure of each letter
   (opener, dateline, salutations, paragraphs, closer, signature, postscript, address) comes out.
2. **Editorial interventions.** Text snippets with bracketed editorial interventions go in
   (`Fr[au]`, `[ein]`, `gefühltes[?]`), the type of each intervention and its TEI encoding come
   out as JSON.

The prompts are from [the Evaluation Framework paper in the *Journal of Open Humanities Data*](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.484)
and its development repository (see Data).

## What you need

- **Session:** JupyterHub, CPU session. No GPU is needed: the model runs behind the cluster's gateway.
- **Resources:** minimal RAM and disk. Each notebook runs in less than a minute on its ten samples (the time depends on the
  load of the shared cluster).
- **Software:** Python 3.9+. The first notebook cell installs `openai`, `python-dotenv`,
  `lxml` and `pandas` from `requirements.txt`.
- **Access:** a project token from [console.dhinfra.uni-graz.at](https://console.dhinfra.uni-graz.at).

## Quickstart

```bash
cd candidates/006-letter-tei-encoding
echo 'DHINFRA_KEY="your-token"' > .env      # stays out of git, see below
```

Then open `01_letters_to_tei.ipynb` (or `02_editorial_interventions.ipynb`) and run all cells.

## What's inside

| File | What it is |
|---|---|
| `01_letters_to_tei.ipynb` | notebook 1: letter structure |
| `02_editorial_interventions.ipynb` | notebook 2: editorial interventions |
| `helpers.py` | code both notebooks import: prompt assembly, the model call, output checks |
| `prompts/letter_structure/` | task, encoding rules, five-shot examples, user message for notebook 1 |
| `prompts/editorial_interventions/` | the same four files for notebook 2 |
| `data/letters/` | ten sample letters as plain text |
| `data/editorial_interventions/sample.json` | ten text snippets with their bracketed sequences |
| `output/` | results, one folder per model |
| `requirements.txt` | Python packages |

## Settings to change before you run

- `DHINFRA_KEY` in `.env` (or the environment): your token. Do not commit the `.env` file.
- `MODEL` in the settings cell of each notebook, if you want another model than
  `deepseek-v4.1-flash`. The notebook prints the models your token can use; the available models
  change, so pick one from that list.
- `DHINFRA_BASE_URL` (optional environment variable): defaults to
  `https://api.dhinfra.uni-graz.at/v1`.

Nothing else points at the author's setup; all paths are relative to the notebooks.

## What it produces

- Notebook 1: `output/<model>/<letter>.xml`, a TEI `<body>` with the letter structure, one file
  per letter. The notebook shows a table with three quick checks: well-formed XML, share of the
  original wording kept, and the elements used.
- Notebook 2: `output/<model>/editorial_interventions.json`, the type and TEI encoding for every
  bracketed sequence. The notebook checks that each sequence was answered, that the type named is one of
  the six defined in the rules, and that the TEI is well-formed and uses only the allowed elements.
  These are format checks: they do not say whether the type or the encoding is correct.

There is no reference encoding in these examples, so look at the results as well.

## Data

**Letters.** Ten letters chosen to cover four languages. They are a slice of the
*Hammer-Purgstall Correspondence TEI Evaluation Dataset* (Sabrina Strutz, Department for Digital
Humanities Graz, <https://zenodo.org/records/17643901>), which holds 100 letters with reference
encodings. The code of the evaluation is at <https://github.com/strubrina/tei-evaluation>.

**Editorial interventions.** Ten text snippets with 1-3 bracketed sequences each, covering all six
intervention types (the text around them is cut at word boundaries), taken from the TEI encoding
of the edition (see below).

**Edition.** The letters (via the Zenodo dataset) and the editorial intervention snippets come from:
Höflechner, Walter; Scholger, Martina; Wagner, Alexandra; Strutz, Sabrina; Steiner, Elisabeth (eds.) 2025: Joseph von Hammer-Purgstall.
Korrespondenz. Graz. <https://gams.uni-graz.at/hpe>, <http://hdl.handle.net/11471/559.20>
(GAMS 559.20).

**Prompts.** `prompts/letter_structure/` is the "five-shot with detailed rules" variant (called
`prompt_v2` in the author's development repository, <https://github.com/strubrina/llm-processing>).
`prompts/editorial_interventions/` is the prompt used there for editorial interventions.

## Licence

- Code (`helpers.py`, the notebooks): Apache-2.0, the repository default.
- Letters, snippets and prompts: CC BY-NC 4.0 (non-commercial use), see `LICENSE.txt`. This is the
  licence of the edition the letters come from (GAMS 559.20) and of the Zenodo dataset above.

## Credits

Letters: Strutz (Zenodo dataset) and Höflechner et al. (edition). Snippets: Höflechner et al. (edition). See Data. Model:
DeepSeek V4.1 Flash, served on the DHinfra cluster (the model used for the sample run).

This work was run on the cluster of Digital Humanities Infrastructure Austria (DHinfra.at). The
GPU systems are described in:

> Atzenhofer-Baumgartner, Florian, David Fleischhacker, Max Resch, Lukas Waldhofer, and
> Michael Otto. 2026. "Design and Operation of a Federated GPU Cluster for Digital
> Humanities within DHinfra.at." arXiv:2609.10552 [cs.DC].
> <https://doi.org/10.48550/arXiv.2609.10552>
