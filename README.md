# Receipt Auditor

Turns receipt photos into reconciled, categorized spending, using a Qwen2.5-1.5B LoRA fine-tune for the categories and plain code for every number.

[Project page](https://lyhjeremy.github.io/receipt-auditor/) · [Write-up](https://lyhjeremy.github.io/receipt-auditor/overview/) · [Engineering notes](writeup.md)

Design and copy follow [these standards](https://github.com/lyhjeremy/lyhjeremy/blob/main/DESIGN_STANDARDS.md).

One of four apps that fine-tune a small local model and benchmark it against Claude, with [Cellar Scanner](https://github.com/lyhjeremy/cellar-scanner), [Race Day Copilot](https://github.com/lyhjeremy/race-day-copilot) and [Menu Decoder](https://github.com/lyhjeremy/menu-decoder).

## Results

12-category spend classifier, held-out test set of 220 synthetic line items from 158 merchants that do not appear in training:

| System | N | Top-1 | Macro-F1 | Latency | Cost/1k calls |
|---|---|---|---|---|---|
| Base Qwen2.5-1.5B | 220 | 27.7% | 0.333 | 5.0 s | $0 |
| Qwen2.5-1.5B + LoRA | 220 | **90.5%** | **0.893** | 4.5 s | $0 |
| Claude, zero-shot | 150 | 56.0% | 0.506 | 17.7 s | Max subscription |

Read the Claude row with care (details under [Limitations](#limitations)). Claude answered in prose that a script had to parse, which probably undercounts it, and it ran with file access inside the repo, so most of its answers had seen the category list. The LoRA figure is exact, and the gap is smaller than 34.5 points.

![Confusion matrix for the LoRA model across 12 categories; the largest errors are household predicted as other, travel as transport, and transport as fuel](eval/confusion_matrix.png)

The adapter gets 199 of the 220 test items right. Household is the weak category (9 of 16, with 7 called `other`); clothing, entertainment, subscriptions, coffee, health and groceries have no errors. The [write-up](https://lyhjeremy.github.io/receipt-auditor/overview/) has the per-category breakdown and every call's latency as interactive charts.

## Method

`scripts/gen_line_items.py` generated 4,799 synthetic line items with `claude -p`, written in point-of-sale styles such as `SBUX #4821`, `WM SUPERCENTER #2847` and `TST* JOE'S PIZZA`. They are split by merchant into 4,309 train, 270 validation and 220 test rows, with a leakage check (`training/prep_categorizer.py`, `eval/dataset_card.json`). The fine-tune is LoRA on `mlx-community/Qwen2.5-1.5B-Instruct-4bit` with MLX on Apple Silicon: batch 4, 12 layers, 800 iterations, learning rate 1.5e-4. Validation loss went from 5.085 to 0.419 (`eval/loss_curve.png`). Base, LoRA and Claude all get the same prompt, merchant and item with no category list; raw generations are in `eval/raw_outputs.json` and scores in `eval/benchmark.md`.

In the app, `tesseract` reads the photo and one `claude -p` call per receipt rejects non-receipts, extracts Pydantic-validated fields and flags low-confidence reads. Line items plus tax are then compared with the printed total, with a tolerance of the larger of $0.02 or 0.5% (`src/reconcile.py`); a mismatch is flagged for human review and left as printed.

Spending questions are parsed into a structured query and answered by pandas (`src/analysis.py`). The LLM only phrases the answer, and `assert_numbers_present` checks that every number survived; if one didn't, a template sentence is used. Repeat questions hit a semantic cache.

For personal data, the schema has no fields for cardholder name, card number, address or loyalty ID, and raw OCR text is discarded. `privacy.sweep_receipt` redacts card-like digit runs, emails and phone numbers and counts the redactions, and hosted sessions stay in memory.

## Stack

Python, Gradio, MLX (`mlx_lm`), Qwen2.5-1.5B-Instruct, Pydantic, pandas, tesseract, Claude via `claude -p`.

## How to run

```bash
pip install -r requirements.txt
./setup_ocr.sh                          # tesseract and language packs, one time
python scripts/gen_line_items.py        # synthetic corpus (long-running, resumable)
python training/prep_categorizer.py     # merchant-split train/valid/test
bash training/lora_harness/train.sh mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    data/lora training/adapters 4 12 800 1.5e-4 768
python app.py
```

No API key is needed. Extraction uses `claude -p` on a Claude subscription. Without a trained adapter, the app's categorizer falls back to `other`. Tests: `pytest tests/` (19 tests).

## Repo layout

| Path | Contents |
|---|---|
| `app.py` | Gradio app: batch upload, reconciled grid, dashboard, voice query |
| `scripts/gen_line_items.py` | Synthetic corpus generator |
| `training/prep_categorizer.py` | Merchant-split, leakage-checked dataset |
| `training/bench_categorizer.py` | Three-way benchmark (reconstructed, see below) |
| `src/reconcile.py` | Totals reconciliation and the narration guard |
| `src/privacy.py` | PII layers |
| `src/analysis.py` | pandas spend queries |
| `src/` | Shared toolkit: llm, vision, guardrails, context, cache |
| `eval/` | Benchmark results, raw outputs, confusion matrix, loss curve, dataset card |

## Limitations

- The data is synthetic only. The spec calls for 10 to 20 hand-labeled receipts of my own as a final test, and none have been added.
- The labels are noisy. They came from the same generator as the training data and are inconsistent in places (a CVS first-aid kit is `other`; Shell unleaded is `fuel` on one row and `transport` on another). The LoRA model learned those habits, and Claude's more reasonable answers on such rows count as errors.
- Claude's answers are prose, and 83 of the 150 mention more than one category, so a script had to pick one. The original rule is lost; the reconstruction takes the longest category name in the answer, which sometimes picks a category Claude was ruling out. 56.0% probably understates Claude.
- Claude saw the categories. `claude -p` ran from inside the repo with file access, and 101 of its 150 answers cite `src/schemas.py`. Menu Decoder fixed the same issue by running from a scratch directory; this benchmark was not rerun.
- The script that produced `eval/benchmark.md` was lost before it was committed. `training/bench_categorizer.py` is a reconstruction that reproduces the LoRA figure exactly and the base and Claude figures approximately (about 26% and 51%).

## License

[MIT](LICENSE) © 2026 Jeremy Lee
