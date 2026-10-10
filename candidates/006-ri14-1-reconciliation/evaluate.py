#!/usr/bin/env python3
"""Score a run against the gold of data/test.jsonl, one decision per mention.

    python evaluate.py --run pilot

Every scorable gold link (a person the index attaches to the digest, with a
mention string in the text) is one item. The system's decision for that
mention is correct when it returns the gold person's id, or "new" for a person
that is not in the catalogue. Results are reported by how well the gold person
is covered in the catalogue, because that is where reconciliation gets hard.
"""
import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUPS = [("known, 10+ catalogue digests", lambda g: g["status"] == "known" and g["catalogue_digests"] >= 10),
          ("known, 5-9", lambda g: g["status"] == "known" and 5 <= g["catalogue_digests"] <= 9),
          ("known, 2-4", lambda g: g["status"] == "known" and 2 <= g["catalogue_digests"] <= 4),
          ("known, 1", lambda g: g["status"] == "known" and g["catalogue_digests"] == 1),
          ("new", lambda g: g["status"] == "new")]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot")
    a = ap.parse_args()
    run_dir = HERE / "runs" / a.run
    preds = {r["id"]: r for r in read_jsonl(run_dir / "predictions.jsonl")}
    gold = {t["id"]: t for t in read_jsonl(HERE / "data" / "test.jsonl") if t["id"] in preds}

    # per group: correct / wrong id / wrongly new / wrongly matched / unsure+error
    cnt = {name: collections.Counter() for name, _ in GROUPS}
    kind = collections.Counter()
    for did, t in gold.items():
        by_mention = {m["mention"]: m for m in preds[did]["mentions"]}
        for g in t["gold"]:
            if g["absent_in_text"] or not g["mention"]:
                continue
            grp = next(name for name, f in GROUPS if f(g))
            m = by_mention.get(g["mention"]["text"])
            if m is None:
                cnt[grp]["missing"] += 1; continue
            if m["decision"] == "match":
                res = "correct" if g["entity_id"] in m["entity_ids"] else ("wrong id" if g["status"] == "known" else "matched a new person")
            elif m["decision"] == "new":
                res = "correct" if g["status"] == "new" else "called new"
            else:
                res = m["decision"] or "error"
            cnt[grp][res] += 1
            kind[(g["mention"]["kind"], res == "correct")] += 1

    cols = ["correct", "wrong id", "called new", "matched a new person", "unsure", "error", "missing"]
    tot = collections.Counter()
    for name, _ in GROUPS:
        tot.update(cnt[name])
    n = sum(tot.values())
    known_n = sum(sum(cnt[g].values()) for g, _ in GROUPS if g != "new")
    known_ok = sum(cnt[g]["correct"] for g, _ in GROUPS if g != "new")
    new_n, new_ok = sum(cnt["new"].values()), cnt["new"]["correct"]
    settings = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").exists() else {}
    tp_m = known_ok; fn_m = known_n - known_ok
    fp_m = sum(cnt[g]["wrong id"] for g, _ in GROUPS if g != "new") + cnt["new"]["matched a new person"]
    tp_n = new_ok; fn_n = new_n - new_ok; fp_n = sum(cnt[g]["called new"] for g, _ in GROUPS if g != "new")
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        return p, r, (2 * p * r / (p + r) if p + r else 0)
    L = [f"# Evaluation of run `{a.run}`", "",
         f"{len(gold)} digests, {n:,} mentions; one decision per person the index attaches to a digest and that "
         f"occurs in its text. A match is correct when it returns that person's catalogue id; a new is correct "
         f"when the person is not in the catalogue.", "",
         "| decision | precision | recall | F1 | n (gold) |", "|---|---|---|---|---|"]
    for name, (p, r, f), k in (("match: known person, right id", prf(tp_m, fp_m, fn_m), known_n),
                               ("new: person not in the catalogue", prf(tp_n, fp_n, fn_n), new_n)):
        L.append(f"| {name} | {p:.0%} | {r:.0%} | {f:.0%} | {k} |")
    L += ["", "| mentions | correct | of | accuracy |", "|---|---|---|---|"]
    for k in ("direct", "indirect"):
        c, w = kind[(k, True)], kind[(k, False)]
        label = "named in the text" if k == "direct" else "referred to by title or description only"
        L.append(f"| {label} | {c} | {c + w} | {c / (c + w):.0%} |" if c + w else f"| {label} | 0 | 0 | - |")
    L.append(f"| all | {tot['correct']} | {n} | {tot['correct'] / n:.0%} |")
    if settings.get("usage"):
        u = settings["usage"]
        L += ["", f"Run: model {settings.get('model')}, thinking {'on, budget ' + str(settings.get('thinking_budget')) if settings.get('thinking') else 'off'}, "
                  f"{settings.get('api_calls', 0):,} API calls, {settings.get('seconds', 0) // 60} min; "
                  f"tokens in {u['prompt_tokens'] / 1e6:.2f} M ({u['cached_tokens'] / max(u['prompt_tokens'], 1):.0%} from the prefix cache), "
                  f"out {u['completion_tokens'] / 1e6:.2f} M ({u['reasoning_tokens'] / 1e6:.2f} M thinking)."]
    L += ["", "## Details", "", "By how well the gold person is covered in the catalogue (that is where reconciliation gets hard):", "",
          "| gold person | n | " + " | ".join(cols) + " | accuracy |", "|---|---|" + "---|" * len(cols) + "---|"]
    for name, _ in GROUPS:
        c = cnt[name]; k = sum(c.values())
        L.append(f"| {name} | {k} | " + " | ".join(str(c[x]) for x in cols) + f" | {c['correct'] / k:.0%} |" if k else f"| {name} | 0 |" + " 0 |" * len(cols) + " - |")
    L.append(f"| all | {n} | " + " | ".join(str(tot[x]) for x in cols) + f" | {tot['correct'] / n:.0%} |")
    (run_dir / "evaluation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
