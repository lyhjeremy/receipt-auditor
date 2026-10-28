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
| Claude, zero-shot | 220 | 80.0% | 0.770 | 3.3 s | Max subscription |

The Claude row is a clean rerun from 2026-10-28 (`eval/claude_rerun/`). `claude -p` (claude-sonnet-5-5) ran from a scratch directory with no repo files and every tool disabled. Its prompt was the training prompt plus the list of the 12 allowed labels, since a zero-shot model has no other way to learn the label set, and it was asked to answer with one label. A scorer that accepts only an exact label (after normalizing case and whitespace) found 176 of 220 right and no invalid answers. The first run, published until 2026-10-28, reported 56.0% on 150 rows; it ran inside the repo with file access and its prose answers were parsed by a heuristic, so that figure is retired. `python3 eval/claude_rerun/score.py` reproduces the scores.

38 of Claude's 44 misses are on rows the generator labeled `other` or `transport` where Claude answered `household`, `health` or `fuel`: paint, screws and HDMI cables labeled `other`, Shell unleaded labeled `transport`. The LoRA model learned those labeling habits from the training data; Claude had no way to.

![Confusion matrix for the LoRA model across 12 categories; the largest errors are household predicted as other, travel as transport, and transport as fuel](eval/confusion_matrix.png)

The adapter gets 199 of the 220 test items right. Household is the weak category (9 of 16, with 7 called `other`); clothing, entertainment, subscriptions, coffee, health and groceries have no errors. The [write-up](https://lyhjeremy.github.io/receipt-auditor/overview/) has the per-category breakdown and every call's latency as interactive charts.

## Method

`scripts/gen_line_items.py` generated 4,799 synthetic line items with `claude -p`, written in point-of-sale styles such as `SBUX #4821`, `WM SUPERCENTER #2847` and `TST* JOE'S PIZZA`. They are split by merchant into 4,309 train, 270 validation and 220 test rows, with a leakage check (`training/prep_categorizer.py`, `eval/dataset_card.json`). The fine-tune is LoRA on `mlx-community/Qwen2.5-1.5B-Instruct-4bit` with MLX on Apple Silicon: batch 4, 12 layers, 800 iterations, learning rate 1.5e-4. Validation loss went from 5.085 to 0.419 (`eval/loss_curve.png`). Base and LoRA get the training prompt, merchant and item with no category list; raw generations are in `eval/raw_outputs.json` and scores in `eval/benchmark.md`. Claude gets the same text plus the 12 labels (`eval/claude_rerun/prompt_template.txt`).

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
| `training/bench_categorizer.py` | Base and LoRA benchmark (reconstructed, see below) |
| `eval/claude_rerun/` | Clean Claude rerun: runner, prompt, raw outputs, scorer, results |
| `src/reconcile.py` | Totals reconciliation and the narration guard |
| `src/privacy.py` | PII layers |
| `src/analysis.py` | pandas spend queries |
| `src/` | Shared toolkit: llm, vision, guardrails, context, cache |
| `eval/` | Benchmark results, raw outputs, confusion matrix, loss curve, dataset card |

## Limitations

- The data is synthetic only. The spec calls for 10 to 20 hand-labeled receipts of my own as a final test, and none have been added.
- The labels are noisy. They came from the same generator as the training data and are inconsistent in places (a CVS first-aid kit is `other`; Shell unleaded is `fuel` on one row and `transport` on another). The LoRA model learned those habits, and Claude's more reasonable answers on such rows count as errors. This accounts for most of the 10.5-point gap.
- The prompts differ. Claude's prompt lists the 12 labels; the LoRA model's does not, because it learned them in training.
- The script that produced `eval/benchmark.md` was lost before it was committed. `training/bench_categorizer.py` is a reconstruction that reproduces the LoRA figure exactly and the base figure approximately (about 26% against the published 27.7%, from prose answers).
- Latency is not like for like. Each local call starts `mlx_lm.generate` and loads the model, which is most of its 4.5 s.
- `app.py` asks the adapter a slightly different question from the training prompt (it appends the category list and "Answer with just the category."), so the app's categorizer is not exactly the benchmarked setup.
- 2026-10-28: the Claude row was rerun cleanly and rescored with an exact-match rule. The first run's 56.0% (150 rows) ran inside the repo, where 101 of its 150 answers cited `src/schemas.py`, and was scored by a lost prose-parsing heuristic. Its raw outputs stay in `eval/raw_outputs.json`.

## License

[MIT](LICENSE) © 2026 Jeremy Lee
