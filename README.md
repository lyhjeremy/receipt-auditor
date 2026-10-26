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

Read the Claude row with care (details under [Limitations](#limitations)): Claude answered in prose that the scorer parsed by substring match, which undercounts it, and it ran with file access inside the repo, so most answers had seen the category list. The LoRA figure is exact; the gap is smaller than 34.5 points.

![Confusion matrix for the LoRA model across 12 categories; the largest errors are household predicted as other, travel as transport, and transport as fuel](eval/confusion_matrix.png)

## Method

- **Data.** 4,799 synthetic line items generated with `claude -p` in real point-of-sale styles (`SBUX #4821`, `WM SUPERCENTER #2847`, `TST* JOE'S PIZZA`). Split by merchant into 4,309 train, 270 validation and 220 test, with a leakage check (`training/prep_categorizer.py`, `eval/dataset_card.json`).
- **Fine-tune.** LoRA on `mlx-community/Qwen2.5-1.5B-Instruct-4bit` with MLX on Apple Silicon: batch 4, 12 layers, 800 iterations, learning rate 1.5e-4. Validation loss went from 5.085 to 0.419 (`eval/loss_curve.png`).
- **Benchmark.** Base, LoRA and Claude on the same prompt (merchant and item, no category list). Raw generations are in `eval/raw_outputs.json`; scores in `eval/benchmark.md`.
- **Extraction.** `tesseract` OCR, then one `claude -p` call per receipt that rejects non-receipts, extracts Pydantic-validated fields and flags low-confidence reads.
- **Reconciliation.** Line items plus tax are compared with the printed total, tolerance the larger of $0.02 or 0.5% (`src/reconcile.py`). A mismatch is flagged for human review, never corrected automatically.
- **Narration guard.** Spending questions are parsed into a structured query and answered by pandas (`src/analysis.py`). The LLM only phrases the answer; `assert_numbers_present` checks every number survived, otherwise a template sentence is used. Repeat questions hit a semantic cache.
- **PII.** The schema has no fields for cardholder name, card number, address or loyalty ID; raw OCR text is discarded; `privacy.sweep_receipt` redacts card-like digit runs, emails and phone numbers and counts the redactions; hosted sessions stay in memory.

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

- **Synthetic data only.** The spec calls for 10 to 20 real, hand-labeled receipts as a final test; none have been added.
- **Noisy labels.** The labels came from the same generator as the training data and are inconsistent in places (a CVS first-aid kit is `other`; Shell unleaded is `fuel` on one row and `transport` on another). The LoRA model learned those habits; Claude's more reasonable answers on such rows count as errors.
- **Claude scoring.** The scorer takes the longest category name found anywhere in Claude's prose answer, and 83 of Claude's 150 answers mention more than one category, so 56.0% understates Claude.
- **Claude saw the categories.** `claude -p` ran from inside the repo with file access; 101 of its 150 answers cite `src/schemas.py`. Menu Decoder fixed the same issue by running from a scratch directory; this benchmark was not rerun.
- **Reconstructed script.** The script that produced `eval/benchmark.md` was lost before it was committed. `training/bench_categorizer.py` reproduces the LoRA figure exactly and the base and Claude figures approximately (about 26% and 51%).

## License

[MIT](LICENSE) © 2026 Jeremy Lee
