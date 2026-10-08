#!/usr/bin/env python3
"""Person reconciliation on RI XIV,1 digests with the DH Infra LLM API.

    python reconcile.py --run pilot                 # the 50 pilot digests
    python reconcile.py --run test --limit 20       # first 20 test digests
    python reconcile.py --run test                  # all 668 test digests

Input: the person mentions of each test digest as the dataset records them
(the curated mention strings of the printed index's links). For each mention
the catalogue is searched by name similarity (rapidfuzz); if the search has
not really found the person, one API call expands the mention to a full name
and the search is repeated; mentions without candidates are "new" without a
call, the others get one judge call. Results: runs/<run>/predictions.jsonl.
"""
import argparse
import json
import math
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rapidfuzz import fuzz, process

from dhinfra import DHInfra, parse_json

HERE = Path(__file__).resolve().parent
NAME_FLOOR, TOP_K = 82, 6
TITLES = {"kg", "kgin", "kr", "ks", "hg", "hgin", "ehg", "ehgin", "gf", "gfin", "mgf", "pfgf", "landgf", "burggf",
          "fst", "frh", "bf", "ebf", "kard", "papst", "abt", "propst", "dr", "mag", "ritter", "meister", "doge",
          "sultan", "prinz", "kfst", "kaiser", "könig", "herzog", "graf", "infant", "dauphin"}
GLUE = {"von", "zu", "zum", "zur", "van", "de", "der", "des", "di", "da", "d", "aus", "le", "la", "del", "della",
        "und", "il", "lo", "genannt", "gen"}


def fold(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class Catalogue:
    """Name index over the catalogue: every name form -> person ids, weighted by rarity."""

    def __init__(self, persons):
        self.persons = {p["id"]: p for p in persons}
        self.forms = {}                      # folded form -> set of ids
        for p in persons:
            for form in self._forms(p):
                self.forms.setdefault(form, set()).add(p["id"])
        self.keys = list(self.forms)
        n = len(persons)
        # rare forms (surnames) weigh more than frequent ones (given names, offices)
        self.weight = {f: math.log(n / len(ids)) for f, ids in self.forms.items()}

    @staticmethod
    def _tokens(s):
        for tok in re.findall(r"[^\s,;:()/]+", s or ""):
            for t in [tok] + tok.split("-"):
                t = t.strip(".'-")
                if len(t) >= 4 and t[:1].isupper() and t.lower() not in TITLES and t.lower() not in GLUE:
                    yield t

    @classmethod
    def _forms(cls, p):
        head = re.sub(r"\(.*?\)", "", p["headword"].split("/")[0]).strip()
        srcs = [head, p["label"]] + p["variants"] + p["aliases"] + p["profile"]["mention_forms"]
        forms = {fold(t) for s in srcs for t in cls._tokens(s)}
        # titles and dignities are name forms too: "Papst", "Ehg", "Kg" distinguish rulers and
        # let title-only references ("der Papst") find their candidates
        forms |= {"#" + fold(t) for t in cls._titles(p["label"])}
        return forms

    @staticmethod
    def _titles(s):
        for tok in re.findall(r"[^\s,;:()/]+", s or ""):
            t = tok.strip(".'-").lower()
            if t in TITLES:
                yield t

    def candidates(self, mention):
        """Persons whose name forms resemble the mention's tokens, ranked by weighted similarity."""
        scores = {}
        for t in self._titles(mention):
            form = "#" + fold(t)
            for pid in self.forms.get(form, ()):
                scores[pid] = scores.get(pid, 0) + self.weight[form]
        for t in self._tokens(mention):
            best = {}
            for form, score, _ in process.extract(fold(t), self.keys, scorer=fuzz.ratio,
                                                  score_cutoff=NAME_FLOOR, limit=30):
                for pid in self.forms[form]:
                    best[pid] = max(best.get(pid, 0), score / 100 * self.weight[form])
            for pid, v in best.items():
                scores[pid] = scores.get(pid, 0) + v
        ranked = sorted(scores, key=lambda pid: (-scores[pid], -fuzz.token_set_ratio(fold(mention), fold(self.persons[pid]["label"]))))
        return [(pid, round(scores[pid], 2)) for pid in ranked[:TOP_K]]

    def render(self, pid):
        p = self.persons[pid]
        pr = p["profile"]
        lines = [f"- id: {pid}", f"  label: {p['label']}"]
        forms = [v for v in p["variants"] + pr["mention_forms"] if v][:8]
        if forms:
            lines.append(f"  spellings seen: {', '.join(forms)}")
        if p["aliases"]:
            lines.append(f"  also indexed as: {'; '.join(p['aliases'][:4])}")
        lines.append(f"  attested: {pr['date_min']} to {pr['date_max']}, {len(pr['digests'])} digest(s), "
                     f"places: {', '.join(pr['places'][:6]) or '-'}")
        for ex in pr["examples"][:2]:
            lines.append(f"  example ({ex['digest_id']}, {ex['date']}): {ex['text'][:220]}")
        return "\n".join(lines)


WEAK_SIMILARITY = 70   # best candidate name similarity below this: the search has not really found the person


def strip_titles(s):
    return " ".join(t for t in re.findall(r"[^\s,;:()/]+", s) if t.strip(".'-").lower() not in TITLES | GLUE)


def name_similarity(cat, mention, pid):
    """How similar the mention is to the candidate's name (not its office-laden label)."""
    p = cat.persons[pid]
    head = re.sub(r"\(.*?\)", "", p["headword"].split("/")[0]).strip()
    given = (p["subentry"] or "").split(",")[0]
    forms = [f for f in [head + " " + given, head] + p["variants"] + p["profile"]["mention_forms"] if f.strip()]
    m = fold(strip_titles(mention)) or fold(mention)
    return max(fuzz.token_set_ratio(m, fold(strip_titles(f)) or fold(f)) for f in forms)


def reconcile_digest(d, cat, api, prompts):
    """Reconcile every mention string the dataset records for this digest."""
    out = {"id": d["id"], "mentions": []}
    for m in d["mentions"]:
        text = m["text"]
        row = {"mention": text, "kind": m["kind"], "candidates": [], "decision": None,
               "entity_ids": [], "reason": None, "expanded": None}
        cands = cat.candidates(text)
        best = max((name_similarity(cat, text, pid) for pid, _ in cands), default=0)
        row["name_similarity"] = best
        if best < WEAK_SIMILARITY:
            # the string search has not found the person: ask the model who is meant, then search again
            content, _ = api.chat(prompts["expand"], f"Digest {d['id']} ({d['date_start']}, {d['place'] or '-'}):\n{d['text']}\n\nMention: \"{text}\"")
            try:
                exp = parse_json(content)
                if isinstance(exp, list):            # model answered with a bare list of names
                    exp = {"persons": exp, "basis": None}
                names = [n for n in exp.get("persons", []) if isinstance(n, str) and n.strip()]
                row["expanded"] = {"persons": names, "basis": exp.get("basis")}
                seen = {pid for pid, _ in cands}
                for n in names:
                    for pid, sc in cat.candidates(n):
                        if pid not in seen:
                            cands.append((pid, sc)); seen.add(pid)
                cands = cands[:10]
            except ValueError as e:
                row["expanded"] = {"error": str(e)}
        row["candidates"] = [{"id": pid, "score": s, "label": cat.persons[pid]["label"]} for pid, s in cands]
        if not cands:
            row["decision"], row["reason"] = "new", "no catalogue candidate above the name floor"
        else:
            user = (f"Digest {d['id']} ({d['date_start']}, {d['place'] or '-'}):\n{d['text']}\n\n"
                    f"Mention: \"{text}\""
                    + (f"\nThe mention probably refers to: {'; '.join(row['expanded']['persons'])}" if row.get("expanded") and row["expanded"].get("persons") else "")
                    + "\n\nCandidates:\n" + "\n".join(cat.render(pid) for pid, _ in cands))
            content, _ = api.chat(prompts["judge"], user)
            try:
                j = parse_json(content)
                if isinstance(j, list):              # model wrapped its object in a list
                    j = next((x for x in j if isinstance(x, dict)), {})
                row["decision"] = j.get("decision")
                ids = j.get("entity_ids") or ([j["entity_id"]] if j.get("entity_id") else [])
                ids = [i for i in ids if i in {pid for pid, _ in cands}] if j.get("decision") == "match" else []
                row["entity_ids"] = ids
                row["reason"] = j.get("reason")
                if j.get("decision") == "match" and not ids:
                    row["decision"], row["reason"] = "unsure", "model returned no id from the shortlist"
            except ValueError as e:
                row["decision"], row["reason"] = "error", str(e)
        out["mentions"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot", help="pilot (50 digests) or test (all), also the run folder name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None,
                    help="digest ids to (re)run; their rows replace the ones in the existing predictions file")
    ap.add_argument("--workers", type=int, default=16, help="parallel requests (the DH Infra docs suggest 16)")
    ap.add_argument("--model", default=None, help="served model slug (default: see dhinfra.MODEL)")
    ap.add_argument("--thinking", choices=["on", "off"], default="on",
                    help="model reasoning before the answer; on by default, bounded by --thinking-budget")
    ap.add_argument("--thinking-budget", type=int, default=4096, help="max reasoning tokens per call")
    ap.add_argument("--answer-tokens", type=int, default=1024, help="room for the answer after the thinking")
    a = ap.parse_args()
    test = read_jsonl(HERE / "data" / "test.jsonl")
    if a.run == "pilot":
        keep = set((HERE / "data" / "pilot_50.txt").read_text(encoding="utf-8").splitlines())
        test = [t for t in test if t["id"] in keep]
    if a.limit:
        test = test[:a.limit]
    if a.only:
        test = [t for t in test if t["id"] in set(a.only)]
    run_dir = HERE / "runs" / a.run
    api = DHInfra(run_dir / "cache", **({"model": a.model} if a.model else {}),
                  enable_thinking=a.thinking == "on", thinking_budget=a.thinking_budget, answer_tokens=a.answer_tokens)
    cat = Catalogue(read_jsonl(HERE / "data" / "catalogue.jsonl"))
    prompts = {k: (HERE / "prompts" / f"{k}.md").read_text(encoding="utf-8") for k in ("judge", "expand")}
    t0 = time.time()
    def safe(d):
        try:
            return reconcile_digest(d, cat, api, prompts)
        except Exception as e:          # keep the run alive; the digest is marked and can be rerun from cache
            return {"id": d["id"], "mentions": [], "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(safe, test))
    failed = [r["id"] for r in results if r.get("error")]
    if failed:
        print(f"WARNING: {len(failed)} digests failed (first: {results[[r['id'] for r in results].index(failed[0])]['error'][:120]}); "
              f"rerun the same command to retry them, finished calls come from the cache")
    if a.only and (run_dir / "predictions.jsonl").exists():
        done = {r["id"]: r for r in read_jsonl(run_dir / "predictions.jsonl")}
        done.update({r["id"]: r for r in results})
        results = list(done.values())
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_m = sum(len(r["mentions"]) for r in results)
    (run_dir / "run.json").write_text(json.dumps({**api.settings(), "digests": len(results), "api_calls": api.calls,
                                                  "cache_hits": api.cache_hits, "usage": api.usage,
                                                  "seconds": round(time.time() - t0)}, indent=1), encoding="utf-8")
    u = api.usage
    print(f"{len(results)} digests, {n_m} mentions, {api.calls} API calls ({api.cache_hits} from cache), "
          f"{time.time() - t0:.0f}s; tokens in {u['prompt_tokens']:,} (cached {u['cached_tokens']:,}), "
          f"out {u['completion_tokens']:,} (thinking {u['reasoning_tokens']:,}) -> {run_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
