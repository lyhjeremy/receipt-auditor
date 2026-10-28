"""Clean zero-shot Claude rerun for the spend-category benchmark (2026-10-28).

The first Claude run (eval/raw_outputs.json, "claude_out") ran `claude -p`
from inside the repo with file access, and most answers cited
src/schemas.py. This rerun isolates the call:

- The 220 test rows and the 12 labels are copied into a fresh temporary
  directory, and `claude -p` runs with that directory as its working
  directory, so no repo file is reachable.
- `--tools ""` disables every built-in tool, `--strict-mcp-config` loads no
  MCP servers, `--setting-sources ""` loads no settings or CLAUDE.md files,
  and `--system-prompt` replaces the Claude Code agent prompt.
- The prompt is the training prompt (merchant plus line item) followed by the
  list of 12 allowed labels and an instruction to answer with one label.
  A zero-shot model needs the label set; the LoRA model learned it in
  training. See prompt_template.txt.

Rows are taken in the same seed-42 shuffle as training/bench_categorizer.py,
so row i here is row i of lora_out in eval/raw_outputs.json.

Calls run one at a time with a pause between them. Output is appended to
raw_outputs.jsonl after every call, so the run resumes where it stopped. If
the CLI reports an error (quota, rate limit, auth), the script stops.

Usage: python3 eval/claude_rerun/run_claude.py [scratch_dir]   (then score.py)
"""
from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEST_PATH = REPO / "data" / "lora" / "test.jsonl"
CLASSES_PATH = REPO / "eval" / "classes.json"
OUT_PATH = HERE / "raw_outputs.jsonl"
MODEL = "sonnet"
SYSTEM = "You are a text classifier. Follow the user's output format exactly."
PAUSE_S = 2.0
PROMPT_RE = re.compile(r"^Merchant: (.*)\. Line item: (.*)\. Which category is this\?$", re.S)


def load_rows(test_path: Path) -> list[dict]:
    rows = []
    for line in test_path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            rows.append({"prompt": rec["messages"][0]["content"],
                         "true_category": rec["messages"][1]["content"].strip()})
    random.Random(42).shuffle(rows)
    return rows


def build_prompt(template: str, training_prompt: str) -> str:
    m = PROMPT_RE.match(training_prompt)
    if not m:
        raise ValueError(f"unexpected prompt shape: {training_prompt!r}")
    return template.format(merchant=m.group(1), item=m.group(2))


def main() -> None:
    # Optional argument: an empty directory outside the repo to run from.
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="receipt_claude_rerun_"))
    work.mkdir(parents=True, exist_ok=True)
    if REPO in work.resolve().parents or work.resolve() == REPO:
        sys.exit("working directory must be outside the repo")
    shutil.copy(TEST_PATH, work / "test.jsonl")
    shutil.copy(CLASSES_PATH, work / "classes.json")
    shutil.copy(HERE / "prompt_template.txt", work / "prompt_template.txt")
    template = (work / "prompt_template.txt").read_text()
    rows = load_rows(work / "test.jsonl")

    done = 0
    if OUT_PATH.exists():
        done = sum(1 for l in OUT_PATH.read_text().splitlines() if l.strip())
    print(f"{len(rows)} rows, {done} already done, working dir {work}")

    args = ["claude", "-p", "--model", MODEL, "--output-format", "json",
            "--tools", "", "--strict-mcp-config", "--setting-sources", "",
            "--no-session-persistence", "--system-prompt", SYSTEM]
    with OUT_PATH.open("a") as out:
        for i in range(done, len(rows)):
            prompt = build_prompt(template, rows[i]["prompt"])
            start = time.time()
            proc = subprocess.run(args, input=prompt, capture_output=True, text=True,
                                  cwd=work, timeout=180)
            latency = time.time() - start
            try:
                resp = json.loads(proc.stdout)
            except json.JSONDecodeError:
                print(f"row {i}: unparseable CLI output, stopping: {proc.stdout[:300]} {proc.stderr[:300]}")
                sys.exit(1)
            if proc.returncode != 0 or resp.get("is_error"):
                print(f"row {i}: CLI error, stopping: {resp.get('result')!r}")
                sys.exit(1)
            rec = {"i": i, "prompt": prompt, "output": resp["result"],
                   "model": ",".join(resp.get("modelUsage", {}).keys()),
                   "latency_s": round(latency, 3)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{i:3d} {rec['output']!r:32} {latency:.1f}s")
            time.sleep(PAUSE_S)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
