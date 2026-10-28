# Clean Claude zero-shot rerun, 2026-10-28

Model: claude-sonnet-5-5 via `claude -p`, run from a scratch directory with tools disabled. Prompt: `prompt_template.txt` (merchant, line item, the 12 allowed labels, answer with one label).

Scoring: exact match to one of 12 labels after lowercasing, trimming and collapsing whitespace; anything else is invalid and wrong.

| System | N | Correct | Top-1 | Macro-F1 | Invalid |
|---|---|---|---|---|---|
| Claude, zero-shot (clean) | 220 | 176 | 80.0% | 0.770 | 0 |
| Claude, first 150 rows only | 150 | 121 | 80.7% | 0.771 | 0 |
| Qwen2.5-1.5B + LoRA (stored outputs, same rule) | 220 | 199 | 90.5% | 0.893 | 0 |

Claude latency: mean 3.323 s, median 2.391 s per call.
Claude and LoRA give the same label on 179 of 220 rows.

## Per category (correct / n)

| Category | N | Claude | LoRA |
|---|---|---|---|
| groceries | 9 | 9 | 9 |
| dining | 13 | 12 | 12 |
| coffee_snacks | 15 | 15 | 15 |
| transport | 15 | 5 | 11 |
| fuel | 14 | 14 | 12 |
| health | 15 | 15 | 15 |
| household | 16 | 16 | 9 |
| clothing | 30 | 30 | 30 |
| entertainment | 26 | 26 | 26 |
| subscriptions_utilities | 23 | 23 | 23 |
| travel | 16 | 11 | 11 |
| other | 28 | 0 | 26 |

## Claude errors (true -> predicted)

- other -> household: 20
- transport -> fuel: 10
- other -> health: 6
- travel -> transport: 4
- dining -> coffee_snacks: 1
- other -> groceries: 1
- other -> travel: 1
- travel -> fuel: 1

## Invalid outputs

None.
