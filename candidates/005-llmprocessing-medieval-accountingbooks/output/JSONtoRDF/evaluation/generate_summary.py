# Script to generate summary of all evaluations of JSON to RDF encoding using LLM for project "Aldersbach digital".
# Generates a consolidated HTML summary table from all evaluation HTML files in the same directory. Groups results by dataset (L343–L346) and model.

# Author: Maximilian Vogeltanz, University of Graz, 2026

import os
import re
import json
import html as html_module
from html.parser import HTMLParser
from collections import defaultdict

# ---------------------------------------------------------------------------
# HTML parser: extracts scores from one evaluation file
# ---------------------------------------------------------------------------

class ScoreParser(HTMLParser):
    """Extracts Overall Score sections from an evaluation HTML file."""

    SECTION_MAP = {
        "Overall Score": "overall",
        "Combined Overall Score": "overall",
        "Overall Structural + Content Fidelity Score": "struct_content",
        "Overall Structural Score": "structural",
        "Overall Content Fidelity Score": "content",
        "Overall Exact Triple String Score": "exact",
    }

    def __init__(self):
        super().__init__()
        self.scores = {}
        self._current_section = None
        self._in_table = False
        self._in_row = False
        self._row_cells = []
        self._current_cell = ""
        self._in_h2 = False
        self._h2_text = ""
        self._in_td_th = False

    def handle_starttag(self, tag, attrs):
        if tag == "h2":
            self._in_h2 = True
            self._h2_text = ""
        elif tag == "table" and self._current_section:
            self._in_table = True
        elif tag == "tr" and self._in_table:
            self._in_row = True
            self._row_cells = []
        elif tag in ("td", "th") and self._in_row:
            self._in_td_th = True
            self._current_cell = ""

    def handle_endtag(self, tag):
        if tag == "h2":
            self._in_h2 = False
            heading = self._h2_text.strip()
            self._current_section = self.SECTION_MAP.get(heading)
            self._in_table = False
        elif tag == "table":
            self._in_table = False
            self._current_section = None
        elif tag == "tr" and self._in_row:
            self._in_row = False
            self._process_row(self._row_cells)
        elif tag in ("td", "th") and self._in_td_th:
            self._in_td_th = False
            self._row_cells.append(self._current_cell.strip())
            self._current_cell = ""

    def handle_data(self, data):
        if self._in_h2:
            self._h2_text += data
        elif self._in_td_th:
            self._current_cell += data

    def _process_row(self, cells):
        if not cells or not self._current_section:
            return
        section = self._current_section
        if section not in self.scores:
            self.scores[section] = {}
        row_type = cells[0].strip()
        if row_type == "Macro-average":
            key = "macro"
        elif row_type == "Micro-average":
            key = "micro"
        else:
            return
        values = [self._parse_float(c) for c in cells[1:]]
        if len(values) == 1:
            self.scores[section][key] = {"f1": values[0]}
        elif len(values) == 3:
            self.scores[section][key] = {
                "precision": values[0], "recall": values[1], "f1": values[2],
            }

    @staticmethod
    def _parse_float(text):
        try:
            return float(text.strip())
        except ValueError:
            return None


def parse_evaluation_file(filepath):
    with open(filepath, encoding="utf-8") as fh:
        content = fh.read()
    parser = ScoreParser()
    parser.feed(content)
    return parser.scores

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_data(directory):
    pattern = re.compile(r"^(L\d+)_(.+)\.html$", re.IGNORECASE)
    data = defaultdict(dict)
    # This repository holds one step, so only this evaluation/ folder is scanned.
    scan_dirs = [directory]
    for scan_dir in scan_dirs:
        if not os.path.isdir(scan_dir):
            continue
        for fname in os.listdir(scan_dir):
            if fname.lower() == "evaluation_jsontordf_total.html":
                continue
            m = pattern.match(fname)
            if not m:
                continue
            dataset_id = m.group(1)
            model_name = m.group(2)
            data[dataset_id][model_name] = parse_evaluation_file(
                os.path.join(scan_dir, fname)
            )
    datasets = sorted(data.keys())
    all_models = set()
    for ds in data.values():
        all_models.update(ds.keys())
    return datasets, sorted(all_models), data

# ---------------------------------------------------------------------------
# Local model classification
# ---------------------------------------------------------------------------

# Models running locally via vLLM on the DH-Infra cluster.
# Add name prefixes or exact names here to classify a model as "local".
LOCAL_MODEL_PREFIXES = [
    "qwen",
    "glm",
    "gemma",
    "kimi",
]

def is_local_model(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(name_lower.startswith(p) for p in LOCAL_MODEL_PREFIXES)

# ---------------------------------------------------------------------------
# Best-value helpers
# ---------------------------------------------------------------------------

def collect_col_values(datasets, models, data, section, avg, metric):
    result = {}
    for ds in datasets:
        for model in models:
            val = data.get(ds, {}).get(model, {}).get(section, {}).get(avg, {}).get(metric)
            result[(ds, model)] = val
    return result

def best_per_dataset(datasets, models, col_values):
    bests = {}
    for ds in datasets:
        vals = [col_values[(ds, m)] for m in models if col_values[(ds, m)] is not None]
        if vals:
            bests[ds] = max(vals)
    return bests

def best_per_dataset_local(datasets, models, col_values):
    """Like best_per_dataset but restricted to local models."""
    local_models = [m for m in models if is_local_model(m)]
    bests = {}
    for ds in datasets:
        vals = [col_values[(ds, m)] for m in local_models if col_values[(ds, m)] is not None]
        if vals:
            bests[ds] = max(vals)
    return bests

def dataset_entry_counts(datasets):
    """Number of encoded entries per dataset, read from the reduced ground truth.

    Used to weight the cross-dataset average by dataset size. Counted at run
    time rather than hard-coded, so the weighting follows the ground truth if
    entries are added or a volume joins. A dataset whose file cannot be read
    falls back to weight 1 with a warning, rather than silently skewing the mean."""
    import xml.etree.ElementTree as ET

    RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
    ENTRY_TAGS = {"Transaction", "Note", "TotalTransaction"}
    # evaluation/ -> JSONtoRDF/ -> output/ -> repository root
    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    gt_dir = os.path.join(repo_dir, "data", "GroundTruth")

    counts = {}
    for ds in datasets:
        path = os.path.join(gt_dir, f"tordf{ds}_RDF-GT.xml")
        try:
            root = ET.parse(path).getroot()
            counts[ds] = sum(
                1 for n in root
                if n.tag.rsplit("}", 1)[-1] in ENTRY_TAGS and n.get(RDF_ABOUT)
            )
        except Exception as exc:
            print(f"  !! entry count for {ds} unavailable ({exc}); using weight 1")
            counts[ds] = 1
    return counts


def compute_model_averages(datasets, models, data):
    """For each model, compute the mean score per (section, avg, metric) across all datasets.
    Only includes models that have data for every dataset in `datasets`.

    The cross-dataset mean is weighted by the number of entries per dataset.
    Because each dataset's macro score is itself the mean over its entries,
    this yields exactly the pooled per-entry mean: every entry counts the same,
    regardless of which volume it comes from. An unweighted mean would give each
    volume a fixed 1/4 share and thereby over-weight the small ones -- L346 has
    38 entries, L343 has 205, so one L346 entry would count five times as much."""
    weights = dataset_entry_counts(datasets)
    print("  cross-dataset weighting by entry count: " +
          ", ".join(f"{ds}={weights.get(ds, 1)}" for ds in datasets))
    result = {}
    for model in models:
        if not all(model in data.get(ds, {}) for ds in datasets):
            continue
        scores = {}
        for sec_key, _, metrics in SECTIONS:
            scores[sec_key] = {}
            for avg_key in ("macro", "micro"):
                scores[sec_key][avg_key] = {}
                for metric in metrics:
                    pairs = [
                        (data[ds][model].get(sec_key, {}).get(avg_key, {}).get(metric),
                         weights.get(ds, 1))
                        for ds in datasets
                    ]
                    pairs = [(v, w) for v, w in pairs if v is not None]
                    total_w = sum(w for _, w in pairs)
                    scores[sec_key][avg_key][metric] = (
                        sum(v * w for v, w in pairs) / total_w if total_w else None
                    )
        result[model] = scores
    return result

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

SECTIONS = [
    ("overall",        "Overall Score",        ["f1"]),
    ("struct_content", "Struct + Content",     ["f1"]),
    ("structural",     "Structural",           ["precision", "recall", "f1"]),
    ("content",        "Content Fidelity",     ["precision", "recall", "f1"]),
    ("exact",          "Exact Triple",         ["precision", "recall", "f1"]),
]

METRIC_LABELS = {"precision": "P", "recall": "R", "f1": "F1"}

def build_column_list(sections):
    cols = []
    for sec_key, sec_label, metrics in sections:
        for avg_key in ("macro", "micro"):
            for metric in metrics:
                cols.append({
                    "idx": len(cols),
                    "section": sec_key,
                    "sec_label": sec_label,
                    "avg": avg_key,
                    "metric": metric,
                })
    return cols

# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def build_header_rows(sections, columns):
    """Three-level thead: section | avg | metric."""
    l1 = [
        '<th rowspan="3" class="col-id">Model</th>',
    ]
    l2, l3 = [], []

    for sec_key, sec_label, _ in sections:
        sec_cols  = [c for c in columns if c["section"] == sec_key]
        sec_idxs  = ",".join(str(c["idx"]) for c in sec_cols)
        l1.append(
            f'<th colspan="{len(sec_cols)}" class="sec-l1"'
            f' data-cols="{sec_idxs}" data-section="{sec_key}">'
            f'{sec_label}</th>'
        )
        for avg_key, avg_label in [("macro", "Macro"), ("micro", "Micro")]:
            avg_cols = [c for c in sec_cols if c["avg"] == avg_key]
            avg_idxs = ",".join(str(c["idx"]) for c in avg_cols)
            l2.append(
                f'<th class="group-l2" colspan="{len(avg_cols)}"'
                f' data-cols="{avg_idxs}" data-section="{sec_key}">'
                f'{avg_label}</th>'
            )
            for c in avg_cols:
                l3.append(
                    f'<th class="group-l3"'
                    f' data-col="{c["idx"]}" data-section="{sec_key}">'
                    f'{METRIC_LABELS[c["metric"]]}</th>'
                )
    return l1, l2, l3


def build_avg_table_rows(models, avg_scores, columns, bests_map_avg, bests_map_avg_local):
    """Build tbody rows for the aggregated table – one row per model."""
    rows = []
    for i, model in enumerate(models):
        parity = "even" if i % 2 == 0 else "odd"
        cells = [
            f'<td class="col-id col-model row-{parity}">{model}</td>'
        ]
        scores = avg_scores[model]
        local = is_local_model(model)
        for c in columns:
            sec, avg, met = c["section"], c["avg"], c["metric"]
            val = scores.get(sec, {}).get(avg, {}).get(met)
            idx = c["idx"]
            if val is None:
                cells.append(
                    f'<td class="score score-missing"'
                    f' data-col="{idx}" data-section="{sec}">–</td>'
                )
            else:
                best_global = bests_map_avg.get((sec, avg, met))
                is_best_global = best_global is not None and abs(val - best_global) < 1e-9
                best_local = bests_map_avg_local.get((sec, avg, met))
                is_best_local = local and best_local is not None and abs(val - best_local) < 1e-9
                if is_best_global:
                    cls = "score score-best"
                elif is_best_local:
                    cls = "score score-best-local"
                else:
                    cls = "score"
                cells.append(
                    f'<td class="{cls}"'
                    f' data-col="{idx}" data-section="{sec}">{val:.3f}</td>'
                )
        rows.append(f'<tr class="row-{parity}">{"".join(cells)}</tr>')
    return rows

def build_row_cells(scores, columns, bests_map, bests_map_local, dataset, model):
    cells = []
    local = is_local_model(model)
    for c in columns:
        sec, avg, met = c["section"], c["avg"], c["metric"]
        val = scores.get(sec, {}).get(avg, {}).get(met)
        idx = c["idx"]
        if val is None:
            cells.append(
                f'<td class="score score-missing"'
                f' data-col="{idx}" data-section="{sec}">–</td>'
            )
        else:
            best_global = bests_map.get((sec, avg, met, dataset))
            is_best_global = best_global is not None and abs(val - best_global) < 1e-9
            best_local = bests_map_local.get((sec, avg, met, dataset))
            is_best_local = local and best_local is not None and abs(val - best_local) < 1e-9
            if is_best_global:
                cls = "score score-best"
            elif is_best_local:
                cls = "score score-best-local"
            else:
                cls = "score"
            cells.append(
                f'<td class="{cls}"'
                f' data-col="{idx}" data-section="{sec}">{val:.3f}</td>'
            )
    return cells

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

CSS = """
*, *::before, *::after { box-sizing: border-box; }

body {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    background: #eef2f5;
    margin: 24px;
    color: #222;
}
h1 { color: #1a3a50; margin-bottom: 4px; font-size: 20px; }
p.subtitle { color: #555; margin-top: 0; margin-bottom: 18px; font-size: 12px; }

/* ---- Controls ----------------------------------------------------------- */
.controls {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    margin-bottom: 18px;
    padding: 12px 16px;
    background: #fff;
    border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}
.ctrl-group {
    display: flex;
    align-items: center;
    gap: 8px;
}
.ctrl-label {
    font-weight: 600;
    color: #2c3e50;
    font-size: 12px;
    margin-right: 2px;
}
fieldset {
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 5px 10px 5px 8px;
    background: #f7f9fb;
    display: flex;
    gap: 10px;
    align-items: center;
}
legend {
    font-weight: 600;
    color: #2c3e50;
    padding: 0 4px;
    font-size: 11px;
    letter-spacing: .03em;
}
label {
    cursor: pointer;
    user-select: none;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 3px;
}
input[type=checkbox] { cursor: pointer; }

.btn-group { display: flex; gap: 6px; }
button {
    padding: 5px 12px;
    background: #2c3e50;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    white-space: nowrap;
    transition: background .15s;
}
button:hover { background: #3d5a72; }
button.btn-secondary {
    background: #546e7a;
}
button.btn-secondary:hover { background: #607d8b; }

/* ---- Table -------------------------------------------------------------- */
.wrapper { overflow-x: auto; }

table {
    border-collapse: collapse;
    white-space: nowrap;
    background: #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}

th, td {
    border: 1px solid #c8d4dc;
    padding: 5px 9px;
    text-align: center;
}

/* Header rows */
th {
    background: #2c3e50;
    color: #fff;
    font-weight: 600;
}
th.col-id   { background: #1a2a38; }
th.group-l2 { background: #3d5470; }
th.group-l3 { background: #516070; font-weight: 400; font-size: 12px; }

/* Section separators – L1 always visible, l2/l3/data managed by JS */
th.sec-l1 {
    border-left: 3px solid #7fb3d3;   /* visible on dark bg */
}
.section-sep {
    border-left: 3px solid #2c3e50 !important;
}

/* Dataset separator rows */
tr.ds-header td {
    background: #d6e8f4;
    font-weight: 700;
    font-size: 13px;
    color: #1a3a50;
    border-top: 3px solid #2c3e50;
    border-bottom: 2px solid #2c3e50;
    text-align: left;
    padding-left: 12px;
    letter-spacing: .04em;
}

/* ID columns */
td.col-id { font-weight: 600; background: #f0f4f7; }
td.col-model { text-align: left; }
td.col-model a { color: #1a6a9a; text-decoration: none; }
td.col-model a:hover { text-decoration: underline; }

/* Score cells */
td.score { font-variant-numeric: tabular-nums; }
td.score-best {
    background: #b8edb0 !important;
    font-weight: 700;
    color: #1a5c14;
}
td.score-best-local {
    background: #fff3b0 !important;
    font-weight: 700;
    color: #7a5c00;
}
td.score-missing { background: #f0f0f0; color: #aaa; }

/* Alternating rows */
tr.row-even td.score, tr.row-even td.score-missing { background: #f7f9fb; }
tr.row-odd  td.score, tr.row-odd  td.score-missing { background: #fff; }
tr.row-even td.score-best, tr.row-odd td.score-best { background: #b8edb0 !important; }
tr.row-even td.score-best-local, tr.row-odd td.score-best-local { background: #fff3b0 !important; }

/* ---- Tabs --------------------------------------------------------------- */
.tab-bar {
    display: flex;
    gap: 4px;
    margin-bottom: 0;
    margin-top: 18px;
}
.tab-btn {
    padding: 7px 22px;
    background: #3d5470;
    color: #fff;
    border: none;
    border-radius: 6px 6px 0 0;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    transition: background .15s;
}
.tab-btn:hover { background: #4a6585; }
.tab-btn.active { background: #1a2a38; }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* ---- Static panels (Prompt / Parameters) -------------------------------- */
.tab-panel {
    padding: 24px;
    background: #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    min-height: 340px;
    font-size: 13px;
    line-height: 1.6;
    color: #222;
}
.tab-panel h2 {
    color: #1a3a50;
    font-size: 15px;
    margin: 0 0 16px 0;
}
.tab-panel pre {
    white-space: pre-wrap;
    background: #f7f9fb;
    border: 1px solid #dce6ed;
    border-radius: 4px;
    padding: 14px 16px;
    margin: 0;
}
.tab-panel ul { padding-left: 22px; margin: 0; }
.tab-panel li { margin-bottom: 4px; }
"""

# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

JS = r"""
// Column metadata injected from Python
const COLUMNS = __COLUMNS_JSON__;

// Maps section key → checkbox id suffix (replacing underscore with hyphen)
const SECTION_CB_MAP = {
    'overall'       : 'cb-sec-overall',
    'struct_content': 'cb-sec-struct-content',
    'structural'    : 'cb-sec-structural',
    'content'       : 'cb-sec-content',
    'exact'         : 'cb-sec-exact',
};

function getState() {
    const state = {
        macro : document.getElementById('cb-macro').checked,
        micro : document.getElementById('cb-micro').checked,
        p     : document.getElementById('cb-p').checked,
        r     : document.getElementById('cb-r').checked,
        f1    : document.getElementById('cb-f1').checked,
    };
    for (const [sec, cbId] of Object.entries(SECTION_CB_MAP)) {
        state['sec_' + sec] = document.getElementById(cbId).checked;
    }
    return state;
}

function isColVisible(col, state) {
    const avgOk = (col.avg === 'macro' && state.macro) ||
                  (col.avg === 'micro' && state.micro);
    const metOk = (col.metric === 'precision' && state.p) ||
                  (col.metric === 'recall'    && state.r) ||
                  (col.metric === 'f1'        && state.f1);
    const secOk = state['sec_' + col.section] !== false;
    return avgOk && metOk && secOk;
}

function updateTable() {
    const state = getState();
    const vis = new Set(COLUMNS.filter(c => isColVisible(c, state)).map(c => c.idx));

    // Individual cells (L3 headers + data cells)
    document.querySelectorAll('[data-col]').forEach(el => {
        el.style.display = vis.has(parseInt(el.dataset.col)) ? '' : 'none';
    });

    // Colspan-bearing headers (L1 section + L2 avg)
    document.querySelectorAll('[data-cols]').forEach(th => {
        const cols = th.dataset.cols.split(',').map(Number);
        const n = cols.filter(i => vis.has(i)).length;
        if (n === 0) {
            th.style.display = 'none';
        } else {
            th.style.display = '';
            th.setAttribute('colspan', n);
        }
    });

    // Dynamic section-separator borders on first visible col per section
    document.querySelectorAll('.section-sep').forEach(el => {
        el.classList.remove('section-sep');
    });
    const sections = [...new Set(COLUMNS.map(c => c.section))];
    for (const sec of sections) {
        const secCols = COLUMNS.filter(c => c.section === sec);
        const first = secCols.find(c => vis.has(c.idx));
        if (first) {
            document.querySelectorAll(`[data-col="${first.idx}"]`).forEach(el => {
                el.classList.add('section-sep');
            });
            // Also mark the first visible L2 header in this section
            const l2s = document.querySelectorAll(`[data-section="${sec}"][data-cols]`);
            let marked = false;
            l2s.forEach(th => {
                if (!marked && th.style.display !== 'none') {
                    th.classList.add('section-sep');
                    marked = true;
                }
            });
        }
    }
}

// Preset buttons
function resetFilters() {
    ['cb-macro','cb-micro','cb-p','cb-r','cb-f1',
     'cb-sec-overall','cb-sec-struct-content','cb-sec-structural',
     'cb-sec-content','cb-sec-exact'
    ].forEach(id => {
        document.getElementById(id).checked = true;
    });
    updateTable();
}
function f1Only() {
    document.getElementById('cb-p').checked  = false;
    document.getElementById('cb-r').checked  = false;
    document.getElementById('cb-f1').checked = true;
    updateTable();
}
function macroOnly() {
    document.getElementById('cb-macro').checked = true;
    document.getElementById('cb-micro').checked = false;
    updateTable();
}
function microOnly() {
    document.getElementById('cb-macro').checked = false;
    document.getElementById('cb-micro').checked = true;
    updateTable();
}
function f1MacroOnly() {
    document.getElementById('cb-macro').checked = true;
    document.getElementById('cb-micro').checked = false;
    document.getElementById('cb-p').checked  = false;
    document.getElementById('cb-r').checked  = false;
    document.getElementById('cb-f1').checked = true;
    updateTable();
}

function switchTab(name) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    document.querySelector('[data-tab="' + name + '"]').classList.add('active');
    updateTable();
}

window.addEventListener('DOMContentLoaded', updateTable);
"""

CONTROLS_HTML = """
<div class="controls">
  <fieldset>
    <legend>Average</legend>
    <label><input type="checkbox" id="cb-macro" checked onchange="updateTable()"> Macro</label>
    <label><input type="checkbox" id="cb-micro" checked onchange="updateTable()"> Micro</label>
  </fieldset>
  <fieldset>
    <legend>Metrics</legend>
    <label><input type="checkbox" id="cb-f1" checked onchange="updateTable()"> F1</label>
    <label><input type="checkbox" id="cb-p"  checked onchange="updateTable()"> Precision (P)</label>
    <label><input type="checkbox" id="cb-r"  checked onchange="updateTable()"> Recall (R)</label>
  </fieldset>
  <fieldset>
    <legend>Sections</legend>
    <label><input type="checkbox" id="cb-sec-overall"        checked onchange="updateTable()"> Overall Score</label>
    <label><input type="checkbox" id="cb-sec-struct-content" checked onchange="updateTable()"> Struct + Content</label>
    <label><input type="checkbox" id="cb-sec-structural"     checked onchange="updateTable()"> Structural</label>
    <label><input type="checkbox" id="cb-sec-content"        checked onchange="updateTable()"> Content Fidelity</label>
    <label><input type="checkbox" id="cb-sec-exact"          checked onchange="updateTable()"> Exact Triple</label>
  </fieldset>
  <div class="btn-group">
    <button onclick="resetFilters()">Show All</button>
    <button class="btn-secondary" onclick="f1Only()">F1 Only</button>
    <button class="btn-secondary" onclick="macroOnly()">Macro Only</button>
    <button class="btn-secondary" onclick="microOnly()">Micro Only</button>
    <button class="btn-secondary" onclick="f1MacroOnly()">F1 Macro</button>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# Static tab content  ← edit these two variables to fill the tabs
# ---------------------------------------------------------------------------

# evaluation/ -> JSONtoRDF/ -> output/ -> repository root
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

_PROMPT_FILE = os.path.join(
    _REPO_DIR, "data", "prompts", "systemprompt_JSONtoRDF.txt"
)
try:
    with open(_PROMPT_FILE, encoding="utf-8") as _fh:
        PROMPT_HTML = "\n".join(l for l in _fh.read().splitlines() if l.strip())
except FileNotFoundError:
    PROMPT_HTML = f"(File not found: {_PROMPT_FILE})"

PARAMETERS_HTML = """\
<ul>
  <li>Temperature: 0</li>
  <li>max_tokens: 32768</li>
  <li>objects_per_batch: 1</li>
</ul>
"""

# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_html(datasets, models, data):
    columns = build_column_list(SECTIONS)

    # Precompute bests per dataset (for detail tab)
    bests_map = {}
    bests_map_local = {}
    for sec_key, _, metrics in SECTIONS:
        for avg_key in ("macro", "micro"):
            for metric in metrics:
                col_vals = collect_col_values(datasets, models, data, sec_key, avg_key, metric)
                per_ds = best_per_dataset(datasets, models, col_vals)
                for ds, best_val in per_ds.items():
                    bests_map[(sec_key, avg_key, metric, ds)] = best_val
                per_ds_local = best_per_dataset_local(datasets, models, col_vals)
                for ds, best_val in per_ds_local.items():
                    bests_map_local[(sec_key, avg_key, metric, ds)] = best_val

    l1, l2, l3 = build_header_rows(SECTIONS, columns)

    # Detail tab: rows grouped by dataset
    rows_html = []
    for ds in datasets:
        ds_models = sorted(data[ds].keys())
        n_cols = 1 + len(columns)
        rows_html.append(
            f'<tr class="ds-header"><td colspan="{n_cols}">{ds}</td></tr>'
        )
        for i, model in enumerate(ds_models):
            scores = data[ds].get(model, {})
            parity = "even" if i % 2 == 0 else "odd"
            cells  = [
                f'<td class="col-id col-model row-{parity}">'
                f'<a href="{ds}_{model}.html" target="_blank">{model}</a></td>'
            ]
            cells.extend(build_row_cells(scores, columns, bests_map, bests_map_local, ds, model))
            rows_html.append(f'<tr class="row-{parity}">{"".join(cells)}</tr>')

    # Avg tab: one row per model, averaged across datasets (only models present in all datasets)
    avg_scores = compute_model_averages(datasets, models, data)
    avg_models = sorted(avg_scores.keys())
    local_avg_models = [m for m in avg_models if is_local_model(m)]
    bests_map_avg = {}
    bests_map_avg_local = {}
    for sec_key, _, metrics in SECTIONS:
        for avg_key in ("macro", "micro"):
            for metric in metrics:
                vals = [
                    avg_scores[m].get(sec_key, {}).get(avg_key, {}).get(metric)
                    for m in avg_models
                ]
                vals = [v for v in vals if v is not None]
                if vals:
                    bests_map_avg[(sec_key, avg_key, metric)] = max(vals)
                local_vals = [
                    avg_scores[m].get(sec_key, {}).get(avg_key, {}).get(metric)
                    for m in local_avg_models
                ]
                local_vals = [v for v in local_vals if v is not None]
                if local_vals:
                    bests_map_avg_local[(sec_key, avg_key, metric)] = max(local_vals)
    avg_rows_html = build_avg_table_rows(avg_models, avg_scores, columns, bests_map_avg, bests_map_avg_local)

    col_json = json.dumps(columns)
    js_code           = JS.replace("__COLUMNS_JSON__", col_json)
    prompt_safe       = html_module.escape(PROMPT_HTML)

    total_files = sum(len(data[ds]) for ds in datasets)

    thead = f"""<thead>
<tr>{"".join(l1)}</tr>
<tr>{"".join(l2)}</tr>
<tr>{"".join(l3)}</tr>
</thead>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLM Evaluation – Overall Scores Summary for JSON to RDF encoding</title>
<style>
{CSS}
</style>
</head>
<body>
<h1>LLM Evaluation – Overall Scores Summary for JSON to RDF encoding</h1>
<p class="subtitle">
  Summary of all Overall Scores from {total_files}&nbsp;evaluation files &middot;
  Datasets:&nbsp;{', '.join(datasets)} &middot;
  <span style="color:#1a5c14;font-weight:700">Green</span> = best overall &middot;
  <span style="color:#7a5c00;font-weight:700">Yellow</span> = best local model (vLLM)
</p>
<details style="margin-bottom:20px; font-family:Arial, sans-serif;">
  <summary style="cursor:pointer; font-size:16px; font-weight:bold;">
    ℹ️ Explanation of Evaluation Metrics
  </summary>
  <div style="margin-top:10px; font-size:14px; line-height:1.5;">
    <h3>Precision and Recall</h3>
    <ul>
      <li><b>Precision (P)</b>: Of all items the system produced, what fraction were correct?</li>
      <li><b>Recall (R)</b>: Of all items in the ground truth, how many did the system reproduce?</li>
      <br/>
      <li><b>Structural precision/recall</b> measure correctness and completeness of RDF triples (graph structure).</li>
      <li><b>Content fidelity precision/recall</b> measure correctness and completeness of literal values, names, amounts, and other identifier strings.</li>
      <li><b>Exact triple match precision/recall</b> measure correctness and completeness of fully specified RDF statements, evaluated as strict string matches of subject–predicate–object triples.</li>
    </ul>
    <h3>Macro- and Micro-Average</h3>
    <ul>
      <li><b>Macro-average</b>: Each entry counts equally, regardless of size (average of per-entry scores).</li>
      <li><b>Micro-average</b>: All items across all entries are pooled, giving larger entries more weight.</li>
      <br/>
      <li><b>Macro-average example</b>: If two entries score 1.0 and 0.0, macro-average F1 = (1.0 + 0.0) / 2 = 0.5.</li>
      <li><b>Micro-average example</b>: If one entry has 100 correct items and another has 0 out of 1, micro recall = 100 / 101 ≈ 0.99.</li>
    </ul>
    --------------------------------------------------------------------------------------------------------------------------------------
  </div>
</details>
{CONTROLS_HTML}
<div class="tab-bar">
  <button class="tab-btn active" data-tab="detail" onclick="switchTab('detail')">Per Dataset</button>
  <button class="tab-btn" data-tab="avg" onclick="switchTab('avg')">All Datasets (average)</button>
  <button class="tab-btn" data-tab="prompt" onclick="switchTab('prompt')">Prompt</button>
  <button class="tab-btn" data-tab="parameters" onclick="switchTab('parameters')">Parameters</button>
</div>
<div id="tab-detail" class="tab-content active wrapper">
<table id="main-table">
{thead}
<tbody>
{"".join(rows_html)}
</tbody>
</table>
</div>
<div id="tab-avg" class="tab-content wrapper">
<table id="avg-table">
{thead}
<tbody>
{"".join(avg_rows_html)}
</tbody>
</table>
</div>
<div id="tab-prompt" class="tab-content">
  <div class="tab-panel">
    <h2>Prompt</h2>
    <pre>{prompt_safe}</pre>
  </div>
</div>
<div id="tab-parameters" class="tab-content">
  <div class="tab-panel">
    <h2>Parameters</h2>
    {PARAMETERS_HTML}
  </div>
</div>
<script>
{js_code}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets, models, data = collect_data(script_dir)

    if not datasets:
        print("No evaluation files found.")
    else:
        print(f"Found datasets: {datasets}")
        for ds in datasets:
            print(f"  {ds}: {sorted(data[ds].keys())}")

        html = generate_html(datasets, models, data)
        out_path = os.path.join(script_dir, "evaluation_jsontordf_total.html")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"\nSummary written to: {out_path}")
