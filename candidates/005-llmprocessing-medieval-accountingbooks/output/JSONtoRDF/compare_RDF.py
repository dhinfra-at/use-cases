# Script that is designed to compare and evaluate two rdf documents in project "Aldersbach digital" and quantify the results in a meaningful way. 
# The first smaller part of the script prints out certain tables in the console, containing basic triple counts.

# the second larger and more important part of the script creates an html diff visually comparing each accounting book entry's rdf-structure in detail.
# It further creates three different F1 Scores for each entry and for all entries in total: 1) Structural Metrics 2) Content Fidelity 3) Exact triple string
# 1. Structural metrics measures similarity and occurence of triples on pure ID basis (Subject, Predicate, any literal)
# 2. Content fidelity metrics measure equivalence of actual strings in identifier/literals.
# 3. Exact triple string metrics measure the entire triple and its literals as one string.
# Also creates an overall score for all three metrics each and and overall score for all three metrics combined (micro-average & macro-average). Each score is weighted equally.
# It further creates a score for structural + content fidelity. Each score is weighted equally.

# Make sure to adapt the file paths and individual parts of the code as you need it. 
# To find all paths in the code simply search for "adapt as needed": You need to set rdf_test_path, rdf_groundtruth_path, combined_html_path and rb_number

# Author: Maximilian Vogeltanz, University of Graz, 2025

from lxml import etree as ET
import os
from rdflib import Graph, Namespace, RDF
import textwrap
from collections import Counter
from rdflib import URIRef, BNode, Literal
import difflib

# ──────────────── Paths ─────────────────
# All paths stay inside this repository. base_dir is output/JSONtoRDF/.
base_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(os.path.dirname(base_dir))
groundtruth_dir = os.path.join(repo_dir, "data", "GroundTruth")

rdf_test_path = os.path.join(base_dir, "postprocessed", "L343GT_qwen3.5-397b.xml") # adapt as needed
rdf_groundtruth_path = os.path.join(groundtruth_dir, "tordfL343_RDF-GT.xml") # adapt as needed
# The report filename must stay "L<book>_<model>.html": generate_summary.py reads the book
# and the model out of it, and does not match a name with the GT suffix in it.
combined_html_path = os.path.join(base_dir, "evaluation", "L343_qwen3.5-397b.html") # adapt as needed
os.makedirs(os.path.dirname(combined_html_path), exist_ok=True)
# assign document numbers (L stands for Latin here, refers to one of the latin accounting books)
rb_number = "L343" # adapt as needed


# ──────────────── RDF Validation & Counting Triples ─────────────────
# --- Load RDF files ---
def load_graph(path):
    g = Graph()
    try:
        g.parse(path, format="xml")
        print(f"✅ Successfully loaded {len(g)} triples from {os.path.basename(path)}")
    except Exception as e:
        print(f"❌ Failed to load {os.path.basename(path)}: {e}")
        return None
    return g

test_graph = load_graph(rdf_test_path)
gt_graph = load_graph(rdf_groundtruth_path)

if not test_graph or not gt_graph:
    exit()


# --- Define namespaces ---
BK = Namespace("https://gams.uni-graz.at/o:depcha.bookkeeping#")  # Adjust if your RDF uses another namespaces
AREC = Namespace("https://gams.uni-graz.at/o:arec.ontology#")
GAMS = Namespace("https://gams.uni-graz.at/o:gams-ontology#")

OBJECT_TYPES = {
    "bk:Transaction": BK.Transaction,
    "arec:Note": AREC.Note,
    "bk:Total": BK.TotalTransaction,
    "bk:Subtotal": BK.SubTotalTransaction,
    "bk:Transfer": BK.Transfer,
    "bk:Money": BK.Money,
    "bk:Commodity": BK.Commodity,
    "bk:Service": BK.Service,
    "bk:Right": BK.Right,
    "arec:Debt": AREC.Debt,
    "bk:Price": BK.Price, #not used for now, but defined for possible later use
}


# --- Count triples per object type ---
def count_triples_by_type(graph, object_types):
    counts = {}
    for label, cls in object_types.items():
        count = sum(1 for _ in graph.subjects(RDF.type, cls))
        counts[label] = count
    return counts

gt_counts = count_triples_by_type(gt_graph, OBJECT_TYPES)
test_counts = count_triples_by_type(test_graph, OBJECT_TYPES)

# --- TOTAL TRIPLE COUNT AND SUMMARY ---
print("\n📊 RDF Triple Comparison Summary")
print("-" * 60)
print(f"{'Type':<20}{'Test File':<10}{'Ground Truth':<15}{'Difference':<10}")
print("-" * 60)

for t in OBJECT_TYPES.keys():
    test_c = test_counts.get(t, 0)
    gt_c = gt_counts.get(t, 0)
    diff = test_c - gt_c
    print(f"{t:<20}{test_c:<10}{gt_c:<15}{diff:<10}")

print("-" * 60)
print(f"{'TOTAL TRIPLES':<20}{len(test_graph):<10}{len(gt_graph):<15}{len(test_graph) - len(gt_graph):<10}")
print("-" * 60)

# calculate overlap in percent
overlap_percent_unfiltered = (len(test_graph) / len(gt_graph) * 100) if len(gt_graph) else 0
print(f"Overlap Triples: {overlap_percent_unfiltered:.2f}%")
print("-" * 60)


# count triples relating to bk:Money
def count_money_relations(graph, BK):
    # Collect all Money subjects
    money_subjects = set(graph.subjects(RDF.type, BK.Money))

    # Count how many of those subjects have bk:unit
    unit_subjects = {s for s, p, o in graph.triples((None, BK.unit, None)) if s in money_subjects}

    # Count how many of those subjects have bk:quantity (or bk:amount)
    quantity_predicates = [BK.quantity, BK.amount]  # cover both possible names
    quantity_subjects = set()
    for pred in quantity_predicates:
        quantity_subjects |= {s for s, p, o in graph.triples((None, pred, None)) if s in money_subjects}

    return {
        "total_money": len(money_subjects),
        "with_unit": len(unit_subjects),
        "with_quantity": len(quantity_subjects),
    }

test_money_stats = count_money_relations(test_graph, BK)
gt_money_stats   = count_money_relations(gt_graph, BK)

print("\n💰 bk:Money Relation Summary")
print("-" * 80)
print(f"{'Metric':<25}{'Test File':<10}{'Ground Truth':<15}{'Diff':<8}")
print("-" * 80)

for key, label in [
    ("total_money", "Total bk:Money"),
    ("with_unit", "… with bk:unit"),
    ("with_quantity", "… with bk:quantity"),
]:
    test_val = test_money_stats[key]
    gt_val   = gt_money_stats[key]
    diff     = test_val - gt_val
    print(f"{label:<25}{test_val:<10}{gt_val:<15}{diff:<8}")

print("-" * 80)

# count triples relating to bk:Commodity|bk:Service|bk:Right
def count_commodity_relations(graph, BK, class_uri, label):
    subjects = set(graph.subjects(RDF.type, class_uri))
    # Property groups to check
    props = {
        "with_unit": BK.unit,
        "with_quantity": BK.quantity,
        "with_classified": BK.classified
    }

    counts = {
        "label": label,
        "total": len(subjects)
    }

    # Count subjects that have each property
    for key, pred in props.items():
        counts[key] = len({s for s, p, o in graph.triples((None, pred, None)) if s in subjects})

    # Also: subjects that have ALL three properties
    unit_subjects = {s for s, _, _ in graph.triples((None, BK.unit, None)) if s in subjects}
    quantity_subjects = {s for s, _, _ in graph.triples((None, BK.quantity, None)) if s in subjects}
    classified_subjects = {s for s, _, _ in graph.triples((None, BK.classified, None)) if s in subjects}

    counts["with_all_three"] = len(unit_subjects & quantity_subjects & classified_subjects)
    return counts

entities = [
    ("bk:Commodity", BK.Commodity, "Commodity"),
    ("bk:Service",   BK.Service,   "Service"),
    ("bk:Right",     BK.Right,     "Right"),
]

print("\n🏷️ Relation Summary for Commodity / Service / Right")
print("-" * 80)
print(f"{'Type':<12}{'Metric':<25}{'Ground Truth':<15}{'Test File':<10}{'Diff':<10}")
print("-" * 80)

for prefix, class_uri, label in entities:
    test_stats = count_commodity_relations(test_graph, BK, class_uri, label)
    gt_stats   = count_commodity_relations(gt_graph, BK, class_uri, label)

    for key, readable in [
        ("total", "Total instances"),
        ("with_unit", "… with bk:unit"),
        ("with_quantity", "… with bk:quantity"),
        ("with_classified", "… with bk:classified"),
    ]:
        t_val = test_stats[key]
        g_val = gt_stats[key]
        diff = t_val - g_val
        print(f"{label:<12}{readable:<30}{t_val:<10}{g_val:<15}{diff:<8}")

print("-" * 80)

# Count how many bk:Transfer instances have each of the following relations: bk:transfers, bk:when, bk:where, bk:to, bk:from, bk:by, bk:credit, bk:debit.
def count_transfer_relations(graph, BK):
    # collect all Transfer subjects
    transfer_subjects = set(graph.subjects(RDF.type, BK.Transfer))
    
    # note: 'from' is a Python keyword, so we access it via BK['from']
    props = {
        "with_transfers": BK.transfers,
        "with_when":       BK.when,
        "with_where":      BK.where,
        "with_to":         BK.to,
        "with_from":       BK["from"],
        "with_by":         BK.by,
        "with_credit":     BK.credit,
        "with_debit":      BK.debit,
    }
    
    # count per‐predicate
    counts = {"total": len(transfer_subjects)}
    # also build a map of sets so we can do the 'all five' check
    subj_sets = {}
    for key, pred in props.items():
        matched = {s for s, p, o in graph.triples((None, pred, None))
                   if s in transfer_subjects}
        counts[key] = len(matched)
        subj_sets[key] = matched
    
    # how many have *all* five relations
    if transfer_subjects:
        all_predicate_subjects = set.intersection(*subj_sets.values())
        counts["with_all_five"] = len(all_predicate_subjects)
    else:
        counts["with_all_five"] = 0
    
    return counts

# compute for test and ground‐truth
test_transfer_stats = count_transfer_relations(test_graph, BK)
gt_transfer_stats   = count_transfer_relations(gt_graph,   BK)

# print table
print("\n🚚 bk:Transfer Relation Summary")
print("-" * 70)
print(f"{'Metric':<20}{'Test File':<10}{'Ground Truth':<15}{'Diff':<8}")
print("-" * 70)
for key, label in [
    ("total",        "Total bk:Transfer"),
    ("with_transfers","… with bk:transfers"),
    ("with_when",     "… with bk:when"),
    ("with_where",    "… with bk:where"),
    ("with_to",       "… with bk:to"),
    ("with_from",     "… with bk:from"),
    ("with_by",     "… with bk:by"),
    ("with_credit",     "… with bk:credit"),
    ("with_debit",     "… with bk:debit")
]:
    t_val = test_transfer_stats[key]
    g_val = gt_transfer_stats[key]
    diff  = t_val - g_val
    print(f"{label:<20}{t_val:<10}{g_val:<15}{diff:<8}")
print("-" * 70)




### PER-ENTITY COMPARISON & HTML DIFF & F1-SCORING ###

# Define Classes 
ENTRY_CLASSES = [
    BK.Transaction,
    AREC.Note,
    BK.TotalTransaction,
    BK.SubTotalTransaction,
]

# --- Extract entry nodes in XML document order --------------------------------
def extract_entries_in_document_order(xml_path):
    """
    Parse the RDF/XML file with lxml to extract entry nodes in document order.
    This preserves the sequential ordering of elements as they appear in the XML,
    regardless of their rdf:type (Transaction, TotalTransaction, etc.).
    Returns a list of URIRef nodes.
    """
    RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    BK_NS = "https://gams.uni-graz.at/o:depcha.bookkeeping#"
    AREC_NS = "https://gams.uni-graz.at/o:arec.ontology#"

    ENTRY_TAGS = {
        f"{{{BK_NS}}}Transaction",
        f"{{{AREC_NS}}}Note",
        f"{{{BK_NS}}}TotalTransaction",
        f"{{{BK_NS}}}SubTotalTransaction",
    }

    tree = ET.parse(xml_path)
    root = tree.getroot()

    entries = []
    for child in root:
        if child.tag in ENTRY_TAGS:
            about = child.get(f"{{{RDF_NS}}}about")
            if about:
                entries.append(URIRef(about))

    return entries

# Pre-compute document-ordered entry lists for both files
gt_entries_ordered = extract_entries_in_document_order(rdf_groundtruth_path)
test_entries_ordered = extract_entries_in_document_order(rdf_test_path)

print(f"\n📄 Document order: {len(gt_entries_ordered)} GT entries, {len(test_entries_ordered)} test entries")

# --- Extract structure per entry ---------------------------------------------

#    Extract all bk:Transaction / arec:Note / bk:TotalTransaction / bk:SubTotalTransaction instances in document order. Each entry stores its transfers and subtriples.
def extract_entry_structure_list(graph, BK):
    structure_list = []
    
    interesting = [
        BK.transfers, BK.when, BK.where, BK.to, BK["from"],
        BK.by, BK.credit, BK.debit
    ]

    for entry_type in ENTRY_CLASSES:
        for e in graph.subjects(RDF.type, entry_type):
            transfers = set(o for _, _, o in graph.triples((e, BK.consistsOf, None)))
            entry_val = None
            for _, _, o in graph.triples((e, BK.entry, None)):
                if isinstance(o, Literal):
                    entry_val = str(o)
                    break

            transfers_data = {}
            for tr in transfers:
                links = set()
                for pred in interesting:
                    for _, _, obj in graph.triples((tr, pred, None)):
                        if isinstance(obj, URIRef):
                            links.add((str(pred), f"uri:{str(obj)}"))
                        elif isinstance(obj, BNode):
                            links.add((str(pred), f"bnode:{str(obj)}"))
                        elif isinstance(obj, Literal):
                            links.add((str(pred), f"lit:{str(obj)}"))
                        else:
                            links.add((str(pred), f"other:{repr(obj)}"))
                transfers_data[tr] = links

            structure_list.append({
                "node": e,
                "type": entry_type.split("#")[-1] if "#" in str(entry_type) else str(entry_type),
                "entry": entry_val,
                "consistsOf": transfers,
                "transfers": transfers_data,
            })

    return structure_list


def compare_structures_by_index(gt_list, test_list):
    """
    Compare entries positionally: 0th ↔ 0th, 1st ↔ 1st, etc.
    """
    rows = []
    total = max(len(gt_list), len(test_list))

    for idx in range(total):
        if idx >= len(gt_list):
            rows.append((idx + 1, "❌ Extra in Test File", "-", len(test_list[idx]["consistsOf"]), test_list[idx]["type"]))
            continue
        if idx >= len(test_list):
            rows.append((idx + 1, "❌ Missing in Test File", len(gt_list[idx]["consistsOf"]), "-", gt_list[idx]["type"]))
            continue

        g = gt_list[idx]
        t = test_list[idx]
        gt_transfers = set(g["transfers"].keys())
        test_transfers = set(t["transfers"].keys())

        # Compare transfer signatures ignoring node identity
        def transfer_signatures(struct):
            return [frozenset(links) for links in struct["transfers"].values()]

        from collections import Counter
        gt_counter = Counter(transfer_signatures(g))
        test_counter = Counter(transfer_signatures(t))

        sig_missing = sum((gt_counter - test_counter).values())
        sig_extra   = sum((test_counter - gt_counter).values())

        if sig_missing == 0 and sig_extra == 0:
            status = "✅ Structure matches"
        else:
            status = f"⚠️ Structure differs (−{sig_missing}, +{sig_extra})"

        rows.append((
            idx + 1,
            status,
            len(gt_transfers),
            len(test_transfers),
            f"{g['type']} ↔ {t['type']}"
        ))

    return rows


def html_diff_trees(gt_lines, test_lines, entry_index):
    """
    Create a side-by-side HTML diff (with full CSS highlighting)
    and return an HTML snippet (not saved individually).
    """
    differ = difflib.HtmlDiff(wrapcolumn=80)
    # Use make_file() instead of make_table() to preserve styles and coloring
    html = differ.make_file(gt_lines, test_lines,
                            fromdesc="Ground Truth", todesc="Test")

    # Strip the <html>, <head>, and <body> wrappers from difflib output
    # so we can safely embed it into one combined file
    start = html.find("<body>") + len("<body>")
    end = html.find("</body>")
    inner_html = html[start:end].strip()

    # Add a section header
    header = f"<h2 style='font-family:sans-serif;'>Entry #{entry_index+1}</h2>"
    return header + inner_html

def _local_name(uri):
    """Return last fragment of URI (# or /)."""
    if not isinstance(uri, str):
        uri = str(uri)
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]

def extract_structural_triples(graph, root_node, BK, max_depth=5, visited=None, counters=None, is_root=True):
    if visited is None:
        visited = set()
    if counters is None:
        counters = {}

    if root_node in visited or max_depth <= 0:
        return []
    visited.add(root_node)

    triples = []
    ignored_preds = {"type"}

    def get_class_name(node):
        types = [o for _, _, o in graph.triples((node, RDF.type, None))]
        if types:
            return _local_name(str(types[0]))
        return "Unknown"

    # --- SUBJECT ID LOGIC ---
    if is_root:
        # entry-level → canonical placeholder (independent of actual ID numbering)
        subj_id = "E"
    else:
        # internal nodes → use class counters
        class_name = get_class_name(root_node)
        c = counters.get(class_name, 0) + 1
        counters[class_name] = c
        subj_id = f"{class_name}#{c}"

    # --- extract triples ---
    for p, o in graph.predicate_objects(root_node):
        pred = _local_name(str(p))
        if pred in ignored_preds:
            continue

        if isinstance(o, Literal):
            triples.append((subj_id, pred, "lit:any"))
        else:
            obj_class = get_class_name(o)
            triples.append((subj_id, pred, obj_class))
            triples.extend(
                extract_structural_triples(graph, o, BK, max_depth - 1, visited, counters, is_root=False)
            )

    return triples



def compare_structures(gt_graph, test_graph, gt_node, test_node, BK):
    """Compute structural precision, recall, and F1 for two entries (duplicate-aware)."""
    gt_triples = Counter(extract_structural_triples(gt_graph, gt_node, BK))
    test_triples = Counter(extract_structural_triples(test_graph, test_node, BK))

    # Multiset (Counter) arithmetic:
    common  = gt_triples & test_triples
    missing = gt_triples - test_triples
    extra   = test_triples - gt_triples

    # Totals as counts of all occurrences
    intersection_count = sum(common.values())
    missing_count      = sum(missing.values())
    extra_count        = sum(extra.values())

    total_gt   = sum(gt_triples.values())
    total_test = sum(test_triples.values())

    precision = intersection_count / total_test if total_test else 0
    recall    = intersection_count / total_gt if total_gt else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "intersection": intersection_count,
        "missing": missing_count,
        "extra": extra_count,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "triples_gt": total_gt,
        "triples_test": total_test,
    }
    

def _shorten_uri(uri):
    """Strip namespace from a URI or BNode ID for cleaner display."""
    if uri.startswith("_:"):
        return uri  # blank node stays as-is
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]

def _canonicalize_id(short_id, root_prefix):
    """Replace the root entry prefix in a shortened ID with canonical 'E'.
    E.g. root_prefix='T4': 'T4' → 'E', 'T4T1' → 'ET1', 'T4T1EA1' → 'ET1EA1'.
    Non-matching IDs (e.g. person names like 'andreemayr') are returned unchanged."""
    if root_prefix and short_id.startswith(root_prefix):
        return "E" + short_id[len(root_prefix):]
    return short_id

def build_tree_lines(graph, node, BK, indent=0, visited=None, max_depth=5):
    """
    Return a list of tree lines, showing:
    - class names (bk:Transaction, etc.)
    - shortened RDF IDs for clarity
    """
    if visited is None:
        visited = set()
    lines = []

    if node in visited or indent > max_depth * 3:
        return lines
    visited.add(node)

    # Determine class of this node
    types = [o for _, _, o in graph.triples((node, RDF.type, None))]
    type_str = f"{_class_prefix(types[0])}:{_local_name(str(types[0]))}" if types else "unknown"

    # Determine readable ID
    if isinstance(node, URIRef):
        node_id = _shorten_uri(str(node))
    elif isinstance(node, BNode):
        node_id = f"_:{str(node)}"
    else:
        node_id = str(node)

    # Root line
    lines.append(
        " " * indent + f"{type_str}  [{node_id}]"
        if type_str != "unknown"
        else " " * indent + f"[unknown]  [{node_id}]"
    )

    ignored_preds = {"type"}

    # Sort for stable output
    for p, o in sorted(graph.predicate_objects(node), key=lambda x: (str(x[0]), str(x[1]))):
        pred = _local_name(str(p))
        if pred in ignored_preds:
            continue

        prefix = " " * (indent + 2) + f"├── {pred} → "

        # --- OBJECT: URI ---
        if isinstance(o, URIRef):
            o_types = [oo for _, _, oo in graph.triples((o, RDF.type, None))]
            display_id = _shorten_uri(str(o))

            if o_types:
                o_class = f"{_class_prefix(o_types[0])}:{_local_name(str(o_types[0]))}"
                lines.append(prefix + f"{o_class} {display_id}")
                if str(o_types[0]).startswith(str(BK)) or str(o_types[0]).startswith(str(AREC)):
                    lines += build_tree_lines(graph, o, BK, indent + 6, visited, max_depth)
            else:
                lines.append(prefix + f"{display_id}")

        # --- OBJECT: Literal ---
        elif isinstance(o, Literal):
            val = str(o).strip().replace("\n", " ")
            val = textwrap.shorten(val, width=600)
            lines.append(" " * (indent + 2) + f"├── {pred}")
            lines.append(" " * (indent + 6) + f"→ \"{val}\"")

        # --- OBJECT: Blank node ---
        elif isinstance(o, BNode):
            lines.append(prefix + f"[BlankNode id=_{str(o)}]")
            lines += build_tree_lines(graph, o, BK, indent + 6, visited, max_depth)

        # Fallback
        else:
            lines.append(prefix + str(o))

    return lines

def extract_identifier_triples(graph, root_node, max_depth=5, visited=None, root_prefix=None):
    """
    Extracts triples where objects are compared by *string value*:
    - literal → literal value
    - URI → shortened URI fragment (canonicalized if root_prefix is set)
    - BNode → its ID
    Produces (predicate, string_value), ignoring structural noise predicates.
    When root_prefix is set, URI IDs starting with the entry prefix are
    canonicalized (e.g. T4T1 → ET1) so that mere numbering shifts don't
    affect the metric.
    """
    if visited is None:
        visited = set()
    if root_node in visited or max_depth <= 0:
        return []
    visited.add(root_node)

    triples = []
    ignored_preds = {"type"}

    for p, o in graph.predicate_objects(root_node):
        pred = _local_name(str(p))

        # SKIP IGNORED PREDICATES
        if pred in ignored_preds:
            continue

        # Literal
        if isinstance(o, Literal):
            val = str(o).strip()
            val = " ".join(val.split())   # normalize whitespace
            triples.append((pred, val))

        # URIRef
        elif isinstance(o, URIRef):
            val = _shorten_uri(str(o))
            if root_prefix:
                val = _canonicalize_id(val, root_prefix)
            triples.append((pred, val))
            triples.extend(
                extract_identifier_triples(graph, o, max_depth - 1, visited, root_prefix=root_prefix)
            )

        # BNode
        elif isinstance(o, BNode):
            val = f"_:{str(o)}"
            triples.append((pred, val))
            triples.extend(
                extract_identifier_triples(graph, o, max_depth - 1, visited, root_prefix=root_prefix)
            )

    return triples

#def is_fuzzy_match(a, b, threshold=0.90):
#    """Return True if strings a and b are at least threshold-similar."""
#    if not a or not b:
#        return False
#    return SequenceMatcher(None, a, b).ratio() >= threshold


def compare_identifiers(gt_graph, test_graph, gt_node, test_node):
    gt_prefix = _shorten_uri(str(gt_node))
    test_prefix = _shorten_uri(str(test_node))
    gt_ids = Counter(extract_identifier_triples(gt_graph, gt_node, root_prefix=gt_prefix))
    test_ids = Counter(extract_identifier_triples(test_graph, test_node, root_prefix=test_prefix))

    common  = gt_ids & test_ids
    missing = gt_ids - test_ids
    extra   = test_ids - gt_ids

    inter = sum(common.values())
    total_gt = sum(gt_ids.values())
    total_test = sum(test_ids.values())

    precision = inter / total_test if total_test else 0
    recall    = inter / total_gt if total_gt else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "inter": inter,
        "missing": sum(missing.values()),
        "extra": sum(extra.values()),
        "missing_items": list(missing.elements()),
        "extra_items": list(extra.elements())
    }
    
    
# --- Exact triple match logic: Renders triples as one string -----
ARROW = " \u2192 "  # " → " 

def _class_prefix(type_uri):
    """Return the display prefix ('bk' or 'arec') for a class URI."""
    return "arec" if str(type_uri).startswith(str(AREC)) else "bk"

def _is_bk_class(graph, node, BK):
    """True if node has an rdf:type in a known modelling namespace (bk or arec)."""
    for _, _, t in graph.triples((node, RDF.type, None)):
        if str(t).startswith(str(BK)) or str(t).startswith(str(AREC)):
            return True
    return False

def _bk_class_name(graph, node):
    """Return 'prefix:LocalName' of the first rdf:type (best-effort)."""
    types = [o for _, _, o in graph.triples((node, RDF.type, None))]
    if types:
        return f"{_class_prefix(types[0])}:{_local_name(str(types[0]))}"
    return "Unknown"

def _render_subject(graph, node, BK, root_prefix=None):
    """Subject rendering: bk:Class [ID] (matches your example style).
    When root_prefix is set, the ID is canonicalized for metrics."""
    node_id = _shorten_uri(str(node)) if isinstance(node, URIRef) else (f"_:{node}" if isinstance(node, BNode) else str(node))
    if root_prefix:
        node_id = _canonicalize_id(node_id, root_prefix)
    if _is_bk_class(graph, node, BK):
        cls = _bk_class_name(graph, node)
        return f"{cls} [{node_id}]"
    return f"[{node_id}]"

def _render_object(graph, obj, BK, root_prefix=None):
    """
    Object rendering:
    - BK-typed URI: bk:Class ID   (no brackets, like 'bk:Transfer T1T1')
    - non-BK URI: shortened fragment (like 'andreemayr')
    - Literal: raw-ish single-line value
    - BNode: _:id
    When root_prefix is set, URI IDs are canonicalized for metrics.
    """
    if isinstance(obj, URIRef):
        oid = _shorten_uri(str(obj))
        if root_prefix:
            oid = _canonicalize_id(oid, root_prefix)
        if _is_bk_class(graph, obj, BK):
            cls = _bk_class_name(graph, obj)
            return f"{cls} {oid}"
        return oid

    if isinstance(obj, BNode):
        return f"_:{str(obj)}"

    if isinstance(obj, Literal):
        # keep it single-line but otherwise strict
        val = str(obj).replace("\n", " ")
        # OPTIONAL: if you want absolutely no normalization, remove .strip()
        val = val.strip()
        return val

    return str(obj)

def extract_exact_rendered_triples(graph, root_node, BK, max_depth=5, visited=None, root_prefix=None):
    """
    Return list of EXACT rendered triple strings:
        "<subj> → <pred> → <obj>"
    Recurses into URI/BNode objects to include their outgoing triples too.
    Uses visited to avoid cycles. Duplicate triples preserved via list.
    When root_prefix is set, IDs are canonicalized for metrics.
    """
    if visited is None:
        visited = set()
    if root_node in visited or max_depth <= 0:
        return []
    visited.add(root_node)

    ignored_preds = {"type"}  # same behavior as your other extractors
    out = []

    subj_str = _render_subject(graph, root_node, BK, root_prefix=root_prefix)

    # stable order helps reproducibility (not required for Counter, but nice)
    for p, o in sorted(graph.predicate_objects(root_node), key=lambda x: (str(x[0]), str(x[1]))):
        pred = _local_name(str(p))
        if pred in ignored_preds:
            continue

        obj_str = _render_object(graph, o, BK, root_prefix=root_prefix)
        out.append(f"{subj_str}{ARROW}{pred}{ARROW}{obj_str}")

        # recurse into graph nodes
        if isinstance(o, (URIRef, BNode)):
            out.extend(extract_exact_rendered_triples(graph, o, BK, max_depth=max_depth - 1, visited=visited, root_prefix=root_prefix))

    return out

def compare_exact_triple_strings(gt_graph, test_graph, gt_node, test_node, BK, max_depth=5):
    gt_prefix = _shorten_uri(str(gt_node))
    test_prefix = _shorten_uri(str(test_node))
    gt = Counter(extract_exact_rendered_triples(gt_graph, gt_node, BK, max_depth=max_depth, root_prefix=gt_prefix))
    te = Counter(extract_exact_rendered_triples(test_graph, test_node, BK, max_depth=max_depth, root_prefix=test_prefix))

    common  = gt & te
    missing = gt - te
    extra   = te - gt

    inter = sum(common.values())
    total_gt = sum(gt.values())
    total_te = sum(te.values())

    precision = inter / total_te if total_te else 0
    recall    = inter / total_gt if total_gt else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "inter": inter,
        "missing": sum(missing.values()),
        "extra": sum(extra.values()),
        "missing_items": list(missing.elements()),
        "extra_items": list(extra.elements()),
        "triples_gt": total_gt,
        "triples_test": total_te,
    }
    
# Return (a_keep, b_keep) containing only mismatching hunks plus `context`, equal lines around each hunk. Inserts a '…' separator between hunks.
def _mismatch_only_lines(a_lines, b_lines, context=1):
    
    sm = difflib.SequenceMatcher(None, a_lines, b_lines)
    opcodes = sm.get_opcodes()

    a_out, b_out = [], []
    SEP = "…"

    def is_change(tag):
        return tag in ("replace", "delete", "insert")

    for k, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        prev_tag = opcodes[k - 1][0] if k > 0 else None
        next_tag = opcodes[k + 1][0] if k < len(opcodes) - 1 else None

        if tag == "equal":
            # Only keep equal lines if they touch a change hunk (context).
            if (prev_tag and is_change(prev_tag)) or (next_tag and is_change(next_tag)):
                # If equal block sits *before* a change, keep last `context` lines.
                # If it sits *after* a change, keep first `context` lines.
                keep_head = (prev_tag and is_change(prev_tag))
                keep_tail = (next_tag and is_change(next_tag))

                chunk_a = a_lines[i1:i2]
                chunk_b = b_lines[j1:j2]

                if keep_head and context > 0:
                    a_out.extend(chunk_a[:context])
                    b_out.extend(chunk_b[:context])

                if keep_tail and context > 0:
                    a_out.extend(chunk_a[-context:])
                    b_out.extend(chunk_b[-context:])

        else:
            # Insert separator if previous emitted line isn’t already a separator
            if a_out and a_out[-1] != SEP:
                a_out.append(SEP)
            if b_out and b_out[-1] != SEP:
                b_out.append(SEP)

            # Add the actual changed hunk
            if tag in ("replace", "delete"):
                a_out.extend(a_lines[i1:i2])
            else:
                # insert: no GT lines in this region; add placeholder for alignment
                a_out.append("[∅]")

            if tag in ("replace", "insert"):
                b_out.extend(b_lines[j1:j2])
            else:
                # delete: no Test lines in this region; add placeholder for alignment
                b_out.append("[∅]")

            # Add separator after hunk (optional, but makes hunks visually distinct)
            a_out.append(SEP)
            b_out.append(SEP)

    # Trim leading/trailing separators
    while a_out and a_out[0] == SEP:
        a_out.pop(0)
    while b_out and b_out[0] == SEP:
        b_out.pop(0)
    while a_out and a_out[-1] == SEP:
        a_out.pop()
    while b_out and b_out[-1] == SEP:
        b_out.pop()

    # If nothing mismatched, return a friendly one-liner (so the section isn’t empty)
    if not a_out and not b_out:
        return ["(no mismatches)"], ["(no mismatches)"]

    return a_out, b_out

# Second diff visualization: only mismatches (+/- small context), side-by-side HTML.
def html_diff_mismatches_only(gt_lines, test_lines, entry_index, context=1):
    gt_filt, test_filt = _mismatch_only_lines(gt_lines, test_lines, context=context)
    html = html_diff_trees(gt_filt, test_filt, entry_index=entry_index)
    return (html)


# Creating diff for visual alignment of both files
# Generate an HTML diff snippet for one entry by index, including structural metrics, and return both HTML and metrics.
def print_entry_trees_side_by_side(gt_graph, test_graph, BK, gt_entries, test_entries, index=0):
    if index >= len(gt_entries) or index >= len(test_entries):
        print(f"Index {index} out of range for available entries (GT: {len(gt_entries)}, Test: {len(test_entries)}).")
        return "", None, None, None

    gt_node = gt_entries[index]
    test_node = test_entries[index]

    # ---- Compute tree lines for visual diff ----
    gt_lines = build_tree_lines(gt_graph, gt_node, BK)
    test_lines = build_tree_lines(test_graph, test_node, BK)
    
    # ---- Compute structural metrics ----
    structural_metrics = compare_structures(gt_graph, test_graph, gt_node, test_node, BK)
    contentfidelity_metrics = compare_identifiers(gt_graph, test_graph, gt_node, test_node)
    
    exacttriplestring_metrics = compare_exact_triple_strings(gt_graph, test_graph, gt_node, test_node, BK, max_depth=5)


    # ---- Format metrics for HTML ----
    score_html = f"""
    --------------------------------------------------------------------------------------------------------------------------------------
    <div style='font-family:sans-serif; margin-bottom:10px;'>
        <strong>Structural Score (Triples with any literal, canonicalized IDs):</strong><br/>
        Precision: <b>{structural_metrics['precision']:.3f}</b> &nbsp;|&nbsp;
        Recall: <b>{structural_metrics['recall']:.3f}</b> &nbsp;|&nbsp;
        F1-score: <b>{structural_metrics['f1']:.3f}</b><br/>
        Triples (GT/Test): {structural_metrics['triples_gt']} / {structural_metrics['triples_test']}<br/>
        Missing: {structural_metrics['missing']} &nbsp;|&nbsp; Extra: {structural_metrics['extra']}
    </div>
    """
    identifier_html = f"""
    <div style='font-family:sans-serif; margin-bottom:10px;'>
        <strong>Content Fidelity Score (String Comparison):</strong><br/>
        Precision: <b>{contentfidelity_metrics['precision']:.3f}</b> &nbsp;|&nbsp;
        Recall: <b>{contentfidelity_metrics['recall']:.3f}</b> &nbsp;|&nbsp;
        F1-score: <b>{contentfidelity_metrics['f1']:.3f}</b><br/>
        Identifiers (GT/Test): {contentfidelity_metrics['inter']} common / 
            {contentfidelity_metrics['missing']} missing / 
            {contentfidelity_metrics['extra']} extra<br/>

        <details style='margin-top:6px;'>
            <summary style='cursor:pointer;'>Show identifier mismatches</summary>
            <div style='margin-left:15px; font-size:13px;'>
                <b>Missing identifiers:</b><br/>
                {"<br/>".join(f"{p}: {v}" for p,v in contentfidelity_metrics['missing_items']) if contentfidelity_metrics['missing_items'] else "None"}<br/><br/>
                <b>Extra identifiers:</b><br/>
                {"<br/>".join(f"{p}: {v}" for p,v in contentfidelity_metrics['extra_items']) if contentfidelity_metrics['extra_items'] else "None"}
            </div>
        </details>
    </div>
    """
    exact_html = f"""
    <div style='font-family:sans-serif; margin-bottom:10px;'>
        <strong>Exact Triple String Score:</strong><br/>
        Precision: <b>{exacttriplestring_metrics['precision']:.3f}</b> &nbsp;|&nbsp;
        Recall: <b>{exacttriplestring_metrics['recall']:.3f}</b> &nbsp;|&nbsp;
        F1-score: <b>{exacttriplestring_metrics['f1']:.3f}</b><br/>
        Triples (GT/Test): {exacttriplestring_metrics['triples_gt']} / {exacttriplestring_metrics['triples_test']}<br/>
        Common: {exacttriplestring_metrics['inter']} &nbsp;|&nbsp; Missing: {exacttriplestring_metrics['missing']} &nbsp;|&nbsp; Extra: {exacttriplestring_metrics['extra']}
        <details style='margin-top:6px;'>
            <summary style='cursor:pointer;'>Show exact-string mismatches</summary>
            <div style='margin-left:15px; font-size:13px;'>
                <b>Missing (GT-only):</b><br/>
                {"<br/>".join(exacttriplestring_metrics['missing_items']) if exacttriplestring_metrics['missing_items'] else "None"}<br/><br/>
                <b>Extra (Test-only):</b><br/>
                {"<br/>".join(exacttriplestring_metrics['extra_items']) if exacttriplestring_metrics['extra_items'] else "None"}
            </div>
        </details>
    </div>
    """

    # ---- Build HTML diff view ----
    html_diff_full = html_diff_trees(gt_lines, test_lines, entry_index=index)
    
    # mismatches-only diff
    html_diff_mismatch = html_diff_mismatches_only(gt_lines, test_lines, entry_index=index, context=1)

    diff_switch_html = f"""
    <div class="diff-full">
    {html_diff_full}
    </div>

    <div class="diff-mismatch">
    {html_diff_mismatch}
    </div>
    """

    # ---- Combine everything into one section ----
    snippet = f"<section>{score_html}{identifier_html}{exact_html}{diff_switch_html}</section>"
    
    # Return both HTML snippet and metrics (for later aggregation)
    return snippet, structural_metrics, contentfidelity_metrics, exacttriplestring_metrics


    
# ──────────────── Combine all diffs into one HTML file ─────────────────

# Compare all available entries automatically (using document order from XML)
total_entries = min(len(gt_entries_ordered), len(test_entries_ordered))
if len(gt_entries_ordered) != len(test_entries_ordered):
    print(f"\n⚠️ Entry count mismatch: GT has {len(gt_entries_ordered)}, Test has {len(test_entries_ordered)}. Comparing first {total_entries} entries.")

all_html_parts = []
structuralcomparison_rows = []
contentfidelity_rows = []
exacttriplestring_rows = []

for i in range(total_entries):

    snippet, structural_metrics, contentfidelity_metrics, exacttriplestring_metrics = print_entry_trees_side_by_side(
        gt_graph, test_graph, BK, gt_entries_ordered, test_entries_ordered, index=i
    )

    # Add HTML
    if snippet:
        all_html_parts.append(snippet)

    # Add structural comparison score
    if structural_metrics:
        structuralcomparison_rows.append((i + 1, structural_metrics))

    # Add content fidelity score
    if contentfidelity_metrics:
        contentfidelity_rows.append((i + 1, contentfidelity_metrics))
    
    # Add exact triple string score
    if exacttriplestring_metrics:
        exacttriplestring_rows.append((i + 1, exacttriplestring_metrics))

# ─── Compute overall macro & micro averages ────────────────────────────────────
overall = {}

# Macro-average (equal weight per entry)
overall["macro_precision"] = sum(m["precision"] for _, m in structuralcomparison_rows) / len(structuralcomparison_rows)
overall["macro_recall"]    = sum(m["recall"] for _, m in structuralcomparison_rows) / len(structuralcomparison_rows)
overall["macro_f1"]        = sum(m["f1"] for _, m in structuralcomparison_rows) / len(structuralcomparison_rows)

# Micro-average (weighted by triple count)
total_inter = sum(m["intersection"] for _, m in structuralcomparison_rows)
total_gt    = sum(m["triples_gt"] for _, m in structuralcomparison_rows)
total_test  = sum(m["triples_test"] for _, m in structuralcomparison_rows)

overall["micro_precision"] = total_inter / total_test if total_test else 0
overall["micro_recall"]    = total_inter / total_gt if total_gt else 0
overall["micro_f1"]        = (
    2 * overall["micro_precision"] * overall["micro_recall"] /
    (overall["micro_precision"] + overall["micro_recall"])
    if (overall["micro_precision"] + overall["micro_recall"]) else 0
)

# ─── Overall Content Fidelity Scores ─────────────────────────────────
overall_id = {}

# Macro average content fidelity
overall_id["macro_precision"] = sum(m["precision"] for _, m in contentfidelity_rows) / len(contentfidelity_rows)
overall_id["macro_recall"]    = sum(m["recall"] for _, m in contentfidelity_rows) / len(contentfidelity_rows)
overall_id["macro_f1"]        = sum(m["f1"] for _, m in contentfidelity_rows) / len(contentfidelity_rows)

# Micro average content fidelity
total_inter_id = sum(m["inter"] for _, m in contentfidelity_rows)
total_missing_id = sum(m["missing"] for _, m in contentfidelity_rows)
total_extra_id = sum(m["extra"] for _, m in contentfidelity_rows)

total_gt_id = total_inter_id + total_missing_id
total_test_id = total_inter_id + total_extra_id

overall_id["micro_precision"] = total_inter_id / total_test_id if total_test_id else 0
overall_id["micro_recall"] = total_inter_id / total_gt_id if total_gt_id else 0
overall_id["micro_f1"] = (
    2 * overall_id["micro_precision"] * overall_id["micro_recall"] /
    (overall_id["micro_precision"] + overall_id["micro_recall"])
    if (overall_id["micro_precision"] + overall_id["micro_recall"]) else 0
)

# ─── Overall Exact Triple String Score ─────────────────────────────────
overall_exact = {}

overall_exact["macro_precision"] = sum(m["precision"] for _, m in exacttriplestring_rows) / len(exacttriplestring_rows)
overall_exact["macro_recall"]    = sum(m["recall"] for _, m in exacttriplestring_rows) / len(exacttriplestring_rows)
overall_exact["macro_f1"]        = sum(m["f1"] for _, m in exacttriplestring_rows) / len(exacttriplestring_rows)

total_inter_ex = sum(m["inter"] for _, m in exacttriplestring_rows)
total_gt_ex    = sum(m["triples_gt"] for _, m in exacttriplestring_rows)
total_test_ex  = sum(m["triples_test"] for _, m in exacttriplestring_rows)

overall_exact["micro_precision"] = total_inter_ex / total_test_ex if total_test_ex else 0
overall_exact["micro_recall"]    = total_inter_ex / total_gt_ex if total_gt_ex else 0
overall_exact["micro_f1"]        = (
    2 * overall_exact["micro_precision"] * overall_exact["micro_recall"] /
    (overall_exact["micro_precision"] + overall_exact["micro_recall"])
    if (overall_exact["micro_precision"] + overall_exact["micro_recall"]) else 0
)

# ─── Overall Structural Score + Content Fidelity Score ───────────────────────────────────────────

weight_half = 0.5   # weight for structural metrics

combined_structural_contentfidelity = {}

# Macro combined F1
combined_structural_contentfidelity["macro_f1"] = (
    weight_half * overall["macro_f1"] +
    (1 - weight_half) * overall_id["macro_f1"]
)

# Micro combined F1
combined_structural_contentfidelity["micro_f1"] = (
    weight_half * overall["micro_f1"] +
    (1 - weight_half) * overall_id["micro_f1"]
)

# ─── Combined Overall Score Structural Score + Content Fidelity Score + Exact Triple String Score ─────────────────────

overall_score = {}

weight_third = 1 / 3  # equal weights

# Macro-average combined F1
overall_score["macro_f1"] = (
    weight_third * overall["macro_f1"] +
    weight_third * overall_id["macro_f1"] +
    weight_third * overall_exact["macro_f1"]
)

# Micro-average combined F1
overall_score["micro_f1"] = (
    weight_third * overall["micro_f1"] +
    weight_third * overall_id["micro_f1"] +
    weight_third * overall_exact["micro_f1"]
)

overall_html = f"""
<details style="margin-bottom:20px; font-family:Arial, sans-serif;">
  <summary style="cursor:pointer; font-size:16px; font-weight:bold;">
    ℹ️ Explanation of Evaluation Metrics
  </summary>
  <div style="margin-top:10px; font-size:14px; line-height:1.5;">
    <h3>Precision and Recall</h3>
    <ul>
      <li><b>Precision</b>: Of all items the system produced, what fraction were correct?</li>
      <li><b>Recall</b>: Of all items in the ground truth, how many did the system reproduce?</li>
      <br/>
      <li><b>Structural precision/recall</b> measure correctness and completeness of RDF triples (graph structure). Uses canonicalized IDs, so that systemic numbering misalignments due to different categorization are not scored wrongly (f.e.: T34 vs N1 and then , T35 vs T34)</li>
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
    <h3>Canonicalized IDs for Structural Scoring</h3>
    <ul>
    <li>Canonicalized IDs are used for Triple based Scoring. That means that each entry ID (T1, T2, TT1, N1, etc.) has been standardized to "E" in the background so that systemic numbering misalignments due to different categorization are not scored wrongly. (f.e.: T34 vs N1 and then , T35 vs T34 but same entry and same encoding).</li>
    <li>Transfers and Economic asset IDs are not canonicalized, cause they count up from 0 in each entry.</li>
    <li>The comparison still shows and highlights the IDs as they are without canonicalization but does not score it based on the ID.</li>
    </ul>
    --------------------------------------------------------------------------------------------------------------------------------------
  </div>
</details>

<h2>Overall Score</h2>
<table border='1' cellpadding='6'>
<tr><th>Type</th><th>Combined F1</th></tr>
<tr>
  <td>Macro-average</td>
  <td>{overall_score['macro_f1']:.3f}</td>
</tr>
<tr>
  <td>Micro-average</td>
  <td>{overall_score['micro_f1']:.3f}</td>
</tr>
</table>
<h2>Overall Structural + Content Fidelity Score</h2>
<table border='1' cellpadding='6'>
<tr><th>Type</th><th>Combined F1</th></tr>
<tr><td>Macro-average</td>
    <td>{combined_structural_contentfidelity['macro_f1']:.3f}</td></tr>
<tr><td>Micro-average</td>
    <td>{combined_structural_contentfidelity['micro_f1']:.3f}</td></tr>
</table>
<br/><br/>
<h2>Overall Structural Score</h2>
<table border='1' cellpadding='6'>
<tr><th>Type</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
<tr><td>Macro-average</td>
    <td>{overall['macro_precision']:.3f}</td>
    <td>{overall['macro_recall']:.3f}</td>
    <td>{overall['macro_f1']:.3f}</td></tr>
<tr><td>Micro-average</td>
    <td>{overall['micro_precision']:.3f}</td>
    <td>{overall['micro_recall']:.3f}</td>
    <td>{overall['micro_f1']:.3f}</td></tr>
</table>
<h2>Overall Content Fidelity Score</h2>
<table border='1' cellpadding='6'>
<tr><th>Type</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
<tr><td>Macro-average</td>
    <td>{overall_id['macro_precision']:.3f}</td>
    <td>{overall_id['macro_recall']:.3f}</td>
    <td>{overall_id['macro_f1']:.3f}</td></tr>
<tr><td>Micro-average</td>
    <td>{overall_id['micro_precision']:.3f}</td>
    <td>{overall_id['micro_recall']:.3f}</td>
    <td>{overall_id['micro_f1']:.3f}</td></tr>
</table>
<h2>Overall Exact Triple String Score</h2>
<table border='1' cellpadding='6'>
<tr><th>Type</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
<tr><td>Macro-average</td>
    <td>{overall_exact['macro_precision']:.3f}</td>
    <td>{overall_exact['macro_recall']:.3f}</td>
    <td>{overall_exact['macro_f1']:.3f}</td></tr>
<tr><td>Micro-average</td>
    <td>{overall_exact['micro_precision']:.3f}</td>
    <td>{overall_exact['micro_recall']:.3f}</td>
    <td>{overall_exact['micro_f1']:.3f}</td></tr>
</table>
<br/><br/>
--------------------------------------------------------------------------------------------------------------------------------------
"""

# Wrap in one HTML page
full_html = f"""
<html>
<head>
<meta charset="utf-8">
<title>Combined RDF Entry Diffs</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
h1 {{ text-align: center; }}
table.diff {{ width: 100%; border-collapse: collapse; margin-bottom: 40px; }}
.diff_header {{ background-color: #f0f0f0; font-weight: bold; }}
.diff_next {{ background-color: #e0e0ff; }}
.diff_add {{ background-color: #aaffaa; }}
.diff_chg {{ background-color: #ffff77; }}
.diff_sub {{ background-color: #ffaaaa; }}

.diff-mismatch {{ display: none; }}
body.mismatch-mode .diff-full {{ display: none; }}
body.mismatch-mode .diff-mismatch {{ display: block; }}
</style>
</head>
<body>

<h1>RDF Comparison for {rb_number} (Ground Truth vs Test)</h1>

<div style="position: fixed; top: 20px; right: 20px; z-index: 2000; background: white; padding: 10px 12px; border-radius: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.2);">
  <button id="toggleMismatch" style="padding:10px 14px; border:1px solid #999; border-radius:8px; background:#fff; cursor:pointer;" onmouseover="this.style.background='#f5f5f5'" onmouseout="this.style.background='#fff'">
    Show mismatches-only (all entries)
  </button>
</div>

<script>
(function () {{
  const btn = document.getElementById("toggleMismatch");
  const key = "rdf_diff_mode_mismatch";

  function setMode(on) {{
    document.body.classList.toggle("mismatch-mode", on);
    btn.textContent = on ? "Show full diff (all entries)" : "Show mismatches-only (all entries)";
    try {{ localStorage.setItem(key, on ? "1" : "0"); }} catch (e) {{}}
  }}

  let saved = "0";
  try {{ saved = localStorage.getItem(key) || "0"; }} catch (e) {{}}
  setMode(saved === "1");

  btn.addEventListener("click", function () {{
    setMode(!document.body.classList.contains("mismatch-mode"));
  }});
}})();
</script>

{overall_html}
{''.join(all_html_parts)}

</body>
</html>
"""


with open(combined_html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"\n✅ All HTML diffs (with colors) combined and saved to:\n{combined_html_path}")