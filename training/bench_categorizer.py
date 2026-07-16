"""3-way benchmark for the spend-category classifier: base model vs
base+LoRA vs Claude zero-shot. See RECEIPT_AUDITOR_SPEC.md §7.2.

RECONSTRUCTED, NOT ORIGINAL -- disclosed honestly, matching this project's
"report gaps, don't hide them" convention (see other honesty notes in this
repo's writeup.md and AI_GAP_PROJECTS_ROADMAP.md).

The script that originally produced eval/benchmark.{json,md} and
eval/confusions.json was lost from disk before ever being committed (only
a stale training/__pycache__/bench_categorizer.cpython-313.pyc survived,
which is how its existence and this file's name are known at all). This
version was rebuilt from scratch against the surviving eval/raw_outputs.json
(the actual, unmodified cached model generations from that original run --
those are real, not reconstructed) and verified as follows:

- Test-set order: eval/confusions.json (LoRA true->predicted pairs) is
  reproduced EXACTLY by shuffling data/lora/test.jsonl with seed 42 before
  pairing against the cached lora_out list -- same convention as Cellar
  Scanner's bench_classifier.py. This confirms the row alignment is right.
- LoRA's headline number (90.5% top-1) is reproduced EXACTLY by this
  script's parser, because LoRA's raw outputs are always a single clean
  class-name token (e.g. "coffee_snacks") -- unambiguous to parse.
- base/claude_teacher's raw outputs are full freeform prose (e.g. "The
  category for this item would be **Men's Clothing**"), and the exact
  heuristic the original script used to extract a label from that prose was
  NOT recoverable. This script's best-effort parser (substring match against
  the known class vocabulary, defaulting unmatched text to "other") gets
  CLOSE to the originally-published figures (base ~26% vs the published
  27.7%; Claude ~51% vs the published 56.0%) but not byte-identical --
  re-running this script is not guaranteed to reproduce eval/benchmark.md's
  exact numbers for those two systems. The committed eval/ files reflect the
  original authentic run (raw_outputs.json proves real model calls
  happened, just parsed by a since-lost script); this script is provided for
  methodology transparency and to regenerate a qualitatively equivalent
  result, not as a guaranteed byte-exact replay.

Full 220-row held-out (by-merchant) test set is small enough to run base/LoRA
on in full (no subsampling, unlike Cellar Scanner's larger 40-class test set).
Claude teacher runs on a 150-row subsample of that same set (disclosed, not
hidden) to keep the benchmark runnable in one sitting.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "lora_harness"))

import llm

TEST_PATH = Path(__file__).resolve().parent.parent / "data" / "lora" / "test.jsonl"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
ADAPTER_PATH = Path(__file__).resolve().parent.parent / "training" / "adapters"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
CLAUDE_SUBSAMPLE = 150  # matches eval/benchmark.json's claude_teacher n=150


def load_classes() -> list[str]:
    return json.loads((EVAL_DIR / "classes.json").read_text())


def load_test_set() -> list[dict]:
    """Shuffled with a fixed seed (42), same convention as Cellar Scanner's
    bench_classifier.py -- test.jsonl rows are entity-grouped (not randomly
    ordered) since prep.py writes them per-entity, so an unshuffled read
    would bias any subsample (Claude's 150-row slice here) toward whichever
    merchants happen to sort first rather than a representative spread."""
    rows = []
    for line in TEST_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prompt = rec["messages"][0]["content"]
        true_category = rec["messages"][1]["content"].strip()
        rows.append({"prompt": prompt, "true_category": true_category})
    random.Random(42).shuffle(rows)
    return rows


def parse_category(text: str, classes: list[str]) -> str:
    """mlx_lm.generate wraps its answer in a `======` stats banner; Claude's
    raw output is a full explanatory sentence with the category in markdown
    (e.g. "falls under the **`health`** category"). Match against the known
    class vocabulary (case-insensitive substring, longest-first so e.g.
    "coffee_snacks" matches before a shorter false-positive) instead of
    exact equality -- same fix pattern as Cellar Scanner's parser bug
    (AI_GAP_PROJECTS_ROADMAP.md §8.5), applied here from the start.

    Unmatched text defaults to "other" (the taxonomy's catch-all) rather
    than the raw sentence -- a raw sentence can never equal a bare class
    label, so leaving it unmatched would silently guarantee a miss on every
    unparseable row instead of making the same call a human skimming the
    same text would.
    """
    body = text.split("==========")[1].strip() if "==========" in text else text.strip()
    sorted_classes = sorted(classes, key=len, reverse=True)
    match = next((c for c in sorted_classes if c.lower() in body.lower()), None)
    return match if match else "other"


def top1_accuracy(predictions: list[str], truths: list[str]) -> float:
    hits = sum(1 for p, t in zip(predictions, truths) if p == t)
    return hits / len(truths) if truths else 0.0


def macro_f1(predictions: list[str], truths: list[str]) -> float:
    labels = set(truths) | set(predictions)
    f1s = []
    for label in labels:
        tp = sum(1 for p, t in zip(predictions, truths) if p == label and t == label)
        fp = sum(1 for p, t in zip(predictions, truths) if p == label and t != label)
        fn = sum(1 for p, t in zip(predictions, truths) if p != label and t == label)
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s) if f1s else 0.0


def run_mlx(prompts: list[str], adapter_path: str | None) -> tuple[list[str], list[float]]:
    outputs, latencies = [], []
    for prompt in prompts:
        args = ["mlx_lm.generate", "--model", BASE_MODEL, "--prompt", prompt, "--max-tokens", "20"]
        if adapter_path:
            args += ["--adapter-path", adapter_path]
        start = time.time()
        proc = subprocess.run(args, capture_output=True, text=True, timeout=90)
        latencies.append(time.time() - start)
        outputs.append(proc.stdout.strip())
    return outputs, latencies


def run_claude(prompts: list[str]) -> tuple[list[str], list[float]]:
    outputs, latencies = [], []
    for prompt in prompts:
        resp = llm.generate(prompt, tier="smart", max_tokens=200)
        outputs.append(resp.text)
        latencies.append(resp.latency_s)
    return outputs, latencies


def confusion_pairs(predictions: list[str], truths: list[str], top_n: int = 10) -> list[tuple]:
    mistakes = Counter((t, p) for t, p in zip(truths, predictions) if t != p)
    return mistakes.most_common(top_n)


def main():
    raw_outputs_path = EVAL_DIR / "raw_outputs.json"
    rows = load_test_set()
    print(f"Held-out test set: {len(rows)} rows")

    prompts = [r["prompt"] for r in rows]
    truths = [r["true_category"] for r in rows]
    claude_rows = rows[:CLAUDE_SUBSAMPLE]
    claude_truths = [r["true_category"] for r in claude_rows]

    # Cache raw generations to disk PER SYSTEM, not just at the end -- a
    # mid-run claude -p rate-limit failure must not discard already-finished
    # base/LoRA outputs (real incident, see roadmap §8.5).
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cached = json.loads(raw_outputs_path.read_text()) if raw_outputs_path.exists() else {}

    def _save(key: str, value) -> None:
        cached[key] = value
        raw_outputs_path.write_text(json.dumps(cached, ensure_ascii=False))

    if "base_out" in cached:
        print("Reusing cached base model outputs")
        base_out, base_lat = cached["base_out"], cached["base_lat"]
    else:
        print("Running base model...")
        base_out, base_lat = run_mlx(prompts, adapter_path=None)
        _save("base_out", base_out)
        _save("base_lat", base_lat)

    if "lora_out" in cached:
        print("Reusing cached base+LoRA outputs")
        lora_out, lora_lat = cached["lora_out"], cached["lora_lat"]
    else:
        print("Running base+LoRA...")
        lora_out, lora_lat = run_mlx(prompts, adapter_path=str(ADAPTER_PATH))
        _save("lora_out", lora_out)
        _save("lora_lat", lora_lat)

    if "claude_out" in cached:
        print("Reusing cached Claude outputs")
        claude_out, claude_lat = cached["claude_out"], cached["claude_lat"]
    else:
        print(f"Running Claude zero-shot on a {len(claude_rows)}-row subsample...")
        claude_out, claude_lat = run_claude([r["prompt"] for r in claude_rows])
        _save("claude_out", claude_out)
        _save("claude_lat", claude_lat)

    classes = load_classes()
    base_preds = [parse_category(o, classes) for o in base_out]
    lora_preds = [parse_category(o, classes) for o in lora_out]
    claude_preds = [parse_category(o, classes) for o in claude_out]

    results = []
    for name, preds, truth_set, lat in [
        ("base", base_preds, truths, base_lat),
        ("lora", lora_preds, truths, lora_lat),
        ("claude_teacher", claude_preds, claude_truths, claude_lat),
    ]:
        results.append({
            "system": name,
            "n": len(truth_set),
            "top1_acc": round(top1_accuracy(preds, truth_set), 4),
            "macro_f1": round(macro_f1(preds, truth_set), 4),
            "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else 0,
            "cost_per_1k_calls_usd": 0 if name != "claude_teacher" else "Max subscription (no per-call API cost)",
        })

    (EVAL_DIR / "benchmark.json").write_text(json.dumps(results, indent=2))

    confusions = confusion_pairs(lora_preds, truths)
    (EVAL_DIR / "confusions.json").write_text(json.dumps(
        [{"true": t, "predicted": p, "count": c} for (t, p), c in confusions], indent=2
    ))

    md_lines = ["# Receipt Auditor -- Spend-Category Classifier Benchmark", "",
                f"Full held-out test set: {len(rows)} rows, held out by merchant (no merchant "
                f"appears in both train and test). base/LoRA evaluated on a {len(rows)}-row "
                f"subsample; Claude teacher on a separate {CLAUDE_SUBSAMPLE}-row subsample "
                "(both disclosed, not hidden). **No real-receipt column**: Jeremy hasn't yet "
                "supplied the 10-20 labeled real receipts the spec calls for "
                "(RECEIPT_AUDITOR_SPEC.md Sec.2) -- these numbers are synthetic-only.", "",
                "| System | N | Top-1 acc | Macro-F1 | Latency (s/item) | Cost/1k |",
                "|---|---|---|---|---|---|"]
    for r in results:
        md_lines.append(
            f"| {r['system']} | {r['n']} | {r['top1_acc']:.1%} | {r['macro_f1']:.3f} | "
            f"{r['mean_latency_s']} | {r['cost_per_1k_calls_usd']} |"
        )
    md_lines += ["", "## Top confusions (LoRA model, true -> predicted)", ""]
    for (t, p), c in confusions:
        md_lines.append(f"- {t} -> {p}: {c} times")

    (EVAL_DIR / "benchmark.md").write_text("\n".join(md_lines))
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
