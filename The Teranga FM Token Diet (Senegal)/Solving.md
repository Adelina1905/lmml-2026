---
title: The Teranga FM Token Diet — Solution
country: Senegal
status: solved
tags:
  - lmml-2026
  - optimization
  - token-budget
  - senegal
---


| Setting | Choices | Meaning |
| --- | --- | --- |
| `tier` | `cheap`, `large` | Selects the system prompt and which facts can appear without assistance. |
| `style` | `verbose`, `terse` | Controls the instruction length and the greeting/closing around the answer. |
| `fewshot` | `true`, `false` | Attaches or removes a fixed block of worked examples. |
| `context` | `true`, `false` | Attaches the question's supplied context snippet, if one exists. |


1. **Quality:** every question's answer must contain every required fact slot.
2. **Budget:** all prompt and completion tokens together must total at most **1,200**.



The important behavior is in `render_completion_body()` and `build_call()`:

- `large` includes every fact slot.
- `cheap` without usable context includes only slots where `large_only` is false.
- `cheap` with a nonempty supplied context snippet includes every slot, including those marked `large_only`.
- Setting `context: true` when a question has no snippet does not provide assistance.


`count_tokens()` uses this regular expression:

```python
r"\w+|[^\w\s]"
```


Switching from `verbose` to `terse` therefore saves **68 tokens per question** while preserving the same included facts.

The examples cost **38 prompt tokens per question**. In the supplied implementation, `fewshot` never changes the generated answer or its coverage. Setting it to `false` saves tokens without affecting quality.


| Question | Cheap without context | Cheap with context | Large without context | Selected |
| --- | --- | --- | --- | --- |
| q02: currency and monetary union | 38 tokens, FAIL | 65 tokens, PASS | 79 tokens, PASS | Cheap with context |
| q04: cash crop and growing region | 42 tokens, FAIL | 96 tokens, PASS | 79 tokens, PASS | Large without context |


$$
2 \times 2 \times 2 \times 2 = 16 \text{ configurations}.
$$

Across 20 questions, that is only **320 evaluations**.

| Question | Topic | Tier | Context | Tokens |
| --- | --- | --- | --- | ---: |
| q01 | Capital and ocean | cheap | false | 40 |
| q02 | Currency and monetary union | cheap | true | 65 |
| q03 | Official and local languages | cheap | false | 37 |
| q04 | Main cash crop and region | large | false | 79 |
| q05 | Rainy and dry seasons | large | false | 78 |
| q06 | Land neighbors | large | false | 74 |
| q07 | Meaning and cultural role of Teranga | large | false | 77 |
| q08 | Independence and first president | large | false | 76 |
| q09 | Staple crops | cheap | false | 36 |
| q10 | Crop rotation and nitrogen | cheap | true | 66 |
| q11 | Groundnut trading city and region | cheap | false | 47 |
| q12 | Planting and harvest seasons | large | false | 77 |
| q13 | Storage contamination | cheap | true | 58 |
| q14 | Groundnut-derived export | cheap | false | 36 |
| q15 | Western ocean and southern neighbor | cheap | false | 43 |
| q16 | Historic race ending in Dakar | cheap | false | 38 |
| q17 | Largest city and economic hub | cheap | true | 66 |
| q18 | Largest ethnic group and population share | cheap | true | 68 |
| q19 | National football team nickname | cheap | false | 35 |
| q20 | Other exports | cheap | true | 64 |
| **Total** | **All 20 questions pass** | | | **1,160** |
