"""Deterministic scorer for the 2026-10-28 clean Claude rerun.

Rule: an answer counts as the category it names only if, after lowercasing,
trimming and collapsing internal whitespace, it is exactly one of the 12
labels in eval/classes.json. Anything else (extra words, punctuation,
markdown, two labels, a refusal) is invalid and scored as wrong. Invalid
answers are listed in results.json and results.md.

The LoRA model's stored outputs (eval/raw_outputs.json, lora_out, same row
order) are scored with the same rule after removing the mlx_lm.generate
stats banner, as a check that the rule reproduces the published 90.5%.

Macro-F1 is the mean F1 over the 12 labels; an invalid answer is a miss for
the true label and a false positive for no label.

Usage: python3 eval/claude_rerun/score.py
Writes eval/claude_rerun/results.json and results.md. Same inputs give
byte-identical outputs.
"""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CLASSES = json.loads((REPO / "eval" / "classes.json").read_text())


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def to_label(text: str) -> str | None:
    t = normalize(text)
    return t if t in CLASSES else None


def load_truths() -> list[dict]:
    rows = []
    for line in (REPO / "data" / "lora" / "test.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            rows.append({"prompt": rec["messages"][0]["content"],
                         "true": rec["messages"][1]["content"].strip()})
    random.Random(42).shuffle(rows)  # same order as training/bench_categorizer.py
    return rows


def macro_f1(preds: list[str | None], truths: list[str]) -> float:
    f1s = []
    for c in CLASSES:
        tp = sum(p == c and t == c for p, t in zip(preds, truths))
        fp = sum(p == c and t != c for p, t in zip(preds, truths))
        fn = sum(p != c and t == c for p, t in zip(preds, truths))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def summarize(preds: list[str | None], truths: list[str]) -> dict:
    n = len(truths)
    correct = sum(p == t for p, t in zip(preds, truths))
    return {"n": n, "correct": correct, "top1_acc": round(correct / n, 4),
            "macro_f1": round(macro_f1(preds, truths), 4),
            "invalid": sum(p is None for p in preds)}


def main() -> None:
    rows = load_truths()
    recs = [json.loads(l) for l in (HERE / "raw_outputs.jsonl").read_text().splitlines() if l.strip()]
    if len(recs) != len(rows):
        raise SystemExit(f"incomplete run: {len(recs)} of {len(rows)} rows; not scoring")
    for r, row in zip(recs, rows):
        assert row["prompt"].split(". Which")[0] in r["prompt"], f"row {r['i']} misaligned"
    truths = [r["true"] for r in rows]
    claude = [to_label(r["output"]) for r in recs]

    raw = json.loads((REPO / "eval" / "raw_outputs.json").read_text())
    lora_text = [o.split("==========")[1] if "==========" in o else o for o in raw["lora_out"]]
    lora = [to_label(t) for t in lora_text]

    lats = [r["latency_s"] for r in recs]
    per_cat = {}
    for c in CLASSES:
        idx = [i for i, t in enumerate(truths) if t == c]
        per_cat[c] = {"n": len(idx), "claude_correct": sum(claude[i] == c for i in idx),
                      "lora_correct": sum(lora[i] == c for i in idx)}
    conf = Counter((t, p if p else "INVALID") for t, p in zip(truths, claude) if t != p)

    results = {
        "date": "2026-10-28",
        "model": sorted({r["model"] for r in recs}),
        "scoring_rule": "exact match to one of 12 labels after lowercasing, trimming and collapsing whitespace; anything else is invalid and wrong",
        "claude": {**summarize(claude, truths),
                   "mean_latency_s": round(statistics.mean(lats), 3),
                   "median_latency_s": round(statistics.median(lats), 3)},
        "claude_first_150": summarize(claude[:150], truths[:150]),
        "lora_same_rule": summarize(lora, truths),
        "claude_lora_agree": sum(a == b for a, b in zip(claude, lora)),
        "per_category": per_cat,
        "claude_confusions": [{"true": t, "predicted": p, "count": c}
                              for (t, p), c in sorted(conf.items(), key=lambda kv: (-kv[1], kv[0]))],
        "invalid_outputs": [{"i": r["i"], "output": r["output"]} for r, p in zip(recs, claude) if p is None],
        "rows": [{"i": i, "true": t, "claude": c, "lora": l} for i, (t, c, l) in enumerate(zip(truths, claude, lora))],
        "claude_latency_s": lats,
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")

    c, l, f = results["claude"], results["lora_same_rule"], results["claude_first_150"]
    md = ["# Clean Claude zero-shot rerun, 2026-10-28", "",
          f"Model: {', '.join(results['model'])} via `claude -p`, run from a scratch directory with tools disabled. "
          "Prompt: `prompt_template.txt` (merchant, line item, the 12 allowed labels, answer with one label).", "",
          f"Scoring: {results['scoring_rule']}.", "",
          "| System | N | Correct | Top-1 | Macro-F1 | Invalid |", "|---|---|---|---|---|---|",
          f"| Claude, zero-shot (clean) | {c['n']} | {c['correct']} | {c['top1_acc']:.1%} | {c['macro_f1']:.3f} | {c['invalid']} |",
          f"| Claude, first 150 rows only | {f['n']} | {f['correct']} | {f['top1_acc']:.1%} | {f['macro_f1']:.3f} | {f['invalid']} |",
          f"| Qwen2.5-1.5B + LoRA (stored outputs, same rule) | {l['n']} | {l['correct']} | {l['top1_acc']:.1%} | {l['macro_f1']:.3f} | {l['invalid']} |",
          "", f"Claude latency: mean {c['mean_latency_s']} s, median {c['median_latency_s']} s per call.",
          f"Claude and LoRA give the same label on {results['claude_lora_agree']} of {c['n']} rows.", "",
          "## Per category (correct / n)", "", "| Category | N | Claude | LoRA |", "|---|---|---|---|"]
    for k, v in per_cat.items():
        md.append(f"| {k} | {v['n']} | {v['claude_correct']} | {v['lora_correct']} |")
    md += ["", "## Claude errors (true -> predicted)", ""]
    md += [f"- {e['true']} -> {e['predicted']}: {e['count']}" for e in results["claude_confusions"]]
    md += ["", "## Invalid outputs", ""]
    md += [f"- row {e['i']}: {e['output']!r}" for e in results["invalid_outputs"]] or ["None."]
    (HERE / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
