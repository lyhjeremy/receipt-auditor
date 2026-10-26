# Receipt Auditor: a finance app where the model never does the arithmetic

Receipt Auditor turns photographed receipts into reconciled, categorized spending. A Qwen2.5-1.5B model with a LoRA adapter, trained on a laptop, puts each line item into one of 12 categories at 90.5% top-1 accuracy on a held-out test set. Claude, asked the same question zero-shot, scored 56.0%. Every number the user sees comes from code, not from a language model.

Design and copy follow [these standards](https://github.com/lyhjeremy/lyhjeremy/blob/main/DESIGN_STANDARDS.md).

## The idea

If you hand a chatbot a pile of receipt photos and ask how much you spent on dining last month, it has two chances to be wrong: reading the numbers and adding them up. Receipt Auditor splits the work so the model never gets the second chance. Totals are reconciled against the printed total in Python. Spending questions are answered by pandas. The model phrases a number it was handed, and a guard checks that the number came through unchanged.

It is one of four apps that each fine-tune a small local model and benchmark it against Claude. The others are [Cellar Scanner](https://github.com/lyhjeremy/cellar-scanner), [Race Day Copilot](https://github.com/lyhjeremy/race-day-copilot) and [Menu Decoder](https://github.com/lyhjeremy/menu-decoder).

## Extraction without a vision API

The project was first planned around Gemini's vision API. Two API keys tested in the first session both authenticated but had no free-tier quota provisioned, which was a project-configuration problem on Google's side rather than a rate limit. Instead of waiting on a third key, I made extraction local: `tesseract` reads the photo, then one `claude -p` call on my Claude subscription checks that the text is a receipt and pulls structured fields. It refuses photos that aren't receipts and flags low-confidence reads.

## Reconciliation and the narration guard

For each receipt, the app adds the line items and tax and compares the sum with the printed total. The tolerance is the larger of 2 cents or 0.5% of the total. A mismatch is shown flagged for a one-tap correction. Nothing adjusts a total to make it agree.

Spending questions work the same way. "How much did I spend on dining" is parsed into a structured `Query` (metric, category, merchant, date range) and answered by pandas. The LLM sees the question and the final number, never the dataframe, and writes one sentence. `assert_numbers_present` then checks that every number it was given appears in that sentence, within a cent ("$42" matches "$42.00"; a changed value does not). If a number was dropped or altered, the app shows a template sentence instead. A spending total the model made up can't reach the screen, because the model never produces the total. Asking the same question twice hits a semantic cache and skips the parse and phrasing calls.

## Privacy in four layers

1. **The schema.** `Receipt` and `LineItem` have no fields for cardholder name, card number, address or loyalty ID. Extraction never asks for them.
2. **Raw OCR text is discarded.** Only the validated `Receipt` object survives extraction. The OCR text is not cached, logged or stored.
3. **A regex sweep.** `privacy.sweep_receipt` runs over every merchant and item string before anything is cached or stored. Long digit runs are masked to the last four, and emails and phone numbers become `[redacted]`. The number of redactions is reported.
4. **Persistence.** `privacy.is_space_environment()` is the one check both the local and hosted paths use. On a hosted Space, receipts live in memory for the session and nothing is written to disk. Locally, a git-ignored ledger is opt-in.

## The fine-tune

The task: given a merchant string and a line-item description, predict one of 12 categories: groceries, dining, coffee_snacks, transport, fuel, health, household, clothing, entertainment, subscriptions_utilities, travel and other.

**Data.** `scripts/gen_line_items.py` generated 4,799 synthetic line items with `claude -p`, varying the abbreviation styles real receipts use (`SBUX #4821`, `WM SUPERCENTER #2847`, `TST* JOE'S PIZZA`), plus full readable names and generic local shops. `training/prep_categorizer.py` splits by merchant, not by row, so no merchant has items on both sides: 2,537 merchants and 4,309 rows in train, 128 and 270 in validation, 158 and 220 in test. The leakage check passed (`eval/dataset_card.json`).

**Training.** LoRA on Qwen2.5-1.5B-Instruct (4-bit) with MLX on Apple Silicon, using the same settings as the other apps in the series: batch 4, 12 layers, 800 iterations, learning rate 1.5e-4. Validation loss fell from 5.085 to 0.419 (`eval/loss_curve.png`).

**Benchmark.** All three systems got the same prompt, merchant plus item with no category list, which is the format the LoRA model was trained on. Base and LoRA ran on all 220 test rows; Claude ran on the first 150 after a seeded shuffle.

| System | N | Top-1 | Macro-F1 | Latency | Cost/1k calls |
|---|---|---|---|---|---|
| Base Qwen2.5-1.5B | 220 | 27.7% | 0.333 | 5.0 s | $0 |
| Qwen2.5-1.5B + LoRA | 220 | **90.5%** | **0.893** | 4.5 s | $0 |
| Claude, zero-shot | 150 | 56.0% | 0.506 | 17.7 s | Max subscription |

Fine-tuning took the base model from 27.7% to 90.5%, a 3.3x lift. The LoRA model is also about four times faster than a `claude -p` call and costs nothing per call.

<p align="center">
  <img src="eval/confusion_matrix.png" alt="Confusion matrix for the LoRA model across 12 categories; the largest errors are household predicted as other, travel predicted as transport, and transport predicted as fuel" width="600">
</p>

The LoRA model's largest errors are household called other (7 times), travel called transport (4) and transport called fuel (4), then other called health (2) and fuel called transport (2). Household and other overlap on generic home goods, and travel, transport and fuel overlap for anything involving a car.

## How much of the gap is real

The 34.5-point gap between the LoRA model and Claude is smaller than it looks, for four reasons I found when I went back through the raw outputs in `eval/raw_outputs.json`.

**The labels are noisy.** They come from the same generator as the training data and are not consistent. A first-aid kit and cold-relief caplets from CVS are both labeled `other`. A Starbucks latte is `dining`. Shell unleaded is `fuel` on one test row and `transport` on another. The LoRA model learned these habits; Claude answered `health`, `coffee_snacks` and `fuel`, which a person would mostly accept, and was marked wrong.

**The scorer undercounts Claude.** Claude answered in full sentences, such as "would fall under the **`health`** category ... distinct from `household` or `other`". The scorer takes the longest category name that appears anywhere in the answer, which here is `household`. 83 of Claude's 150 answers mention more than one category.

**Claude saw the category list.** The prompt had none, but `claude -p` ran from inside the repo with file access, and 101 of its 150 answers cite `src/schemas.py`. Menu Decoder hit the same problem and fixed it by running the CLI from a scratch directory. This benchmark has not been rerun that way.

**The scoring script is a reconstruction.** The script that produced `eval/benchmark.md` was lost before it was committed. `training/bench_categorizer.py` was rebuilt against the stored raw outputs. It reproduces the LoRA figure exactly, because those outputs are single category names, and it reproduces the test-set order exactly, because the confusion pairs match. For base and Claude, whose outputs are prose, it gets about 26% and 51% against the published 27.7% and 56.0%.

There is also no real-receipt test set. The spec calls for 10 to 20 of my own receipts, photographed and hand-labeled. Claude both labeled the training data and is a column in the benchmark, so a test on the same synthetic distribution can't rule out that real receipts are harder. Until that set exists, every figure here is about synthetic data.

What the result does support is narrower: a 1.5B model with a small adapter learns a fixed 12-label scheme well enough to replace a frontier-model call for this step, at a quarter of the latency and no per-call cost.

## What building it taught me

- **The tests caught two bugs before anything shipped.** A field named `date` on the `Receipt` schema risked shadowing Python's `datetime.date`, which breaks Pydantic model construction when both names are in the same scope. And the narration guard's number regex could match a bare comma left after a number (`"...$120.00, up from..."`) and crash on `float("")`. A test sentence with several comma-separated numbers caught the second one. Both are fixed, and 19 tests pass across reconciliation, privacy and analysis.
- **A benchmark timeout was another program's CPU load.** The first benchmark run hit repeated 90-second timeouts on `mlx_lm.generate` calls that normally take a few seconds. Checking system load (`uptime`, `ps aux` sorted by CPU) showed two unrelated VS Code helper processes at 97% and 65% CPU on a 10-core machine. Raising the subprocess timeout to 240 seconds let the run finish without changing any code.
- **A training log came out empty.** `train.sh` pipes through `tee` to `training/adapters/train.log` for the loss-curve script, but the file was 0 bytes even though training finished, probably because of the Google Drive-synced folder. The same output had gone to an outer run log through a different redirect, so the loss curve was rebuilt from that.
- **Read the raw outputs before trusting a benchmark.** The noisy labels, the scoring problem and the schema leak in the previous section all came from reading `eval/raw_outputs.json` row by row, well after the headline number was published.

---

Built by [Jeremy Lee](https://github.com/lyhjeremy). Code, benchmark and dataset card: [github.com/lyhjeremy/receipt-auditor](https://github.com/lyhjeremy/receipt-auditor).
