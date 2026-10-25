# Results notes

These are the numbers saved from the original local runs. They were checked against the available response logs when the repository was cleaned up.

## Saved metrics

| Condition | Keyword refusals | Refusal rate | ASR heuristic |
| --- | ---: | ---: | ---: |
| Baseline | 60 / 80 | 75.00% | 25.00% |
| Abliterated / HF | 27 / 80 | 33.75% | 63.75% |
| Abliterated / Ollama | 28 / 80 | 35.00% | 65.00% |

The ASR heuristic counts 20, 51, and 52 responses respectively. Rates have not been changed. The CSV column was renamed from `asr` to `asr_heuristic`, and the old machine-specific model path was replaced with a readable label.

## How the metrics work

**Refusal rate** checks responses against an English keyword list. It can miss refusals phrased differently or count an ordinary answer containing a phrase such as `i must`.

**ASR heuristic** counts a response when it has no refusal keyword and more than 50 stripped characters. It does not check relevance or harmfulness. A long off-topic answer can pass that rule.

The project does not report human-scored harmful compliance, general model capability, or answer-quality scores. Harmless responses were generated for inspection, but that alone is not a quality benchmark.

## Input differences

The local files contain 500 harmful and 500 harmless prompts with `prompt,kategori` columns. A separate holdout has 100 prompts per class. Exact-text checks found no overlap between the holdout and the main CSVs, but the saved evaluations did not use that holdout.

| Comparison | Matching prompts |
| --- | ---: |
| Baseline vs HF evaluation | 10 of 80 |
| Baseline vs Ollama evaluation | 80 of 80, same order |
| HF vs Ollama evaluation | 10 of 80 |
| Baseline evaluation vs first 256 harmful rows | 40 of 80 |
| HF evaluation vs first 256 harmful rows | 36 of 80 |
| Ollama evaluation vs first 256 harmful rows | 40 of 80 |

The extraction notebooks use the first 256 rows per class, so some evaluated prompts also appear in the extraction input. Screening uses a subset named `test` too. These checks compare the files available now, not the complete history of every run.

The old loader mentioned AdvBench and Alpaca, but the exact origins and changes to the local CSVs were not recorded. Their names do not establish that the files match those public datasets. Using different CSVs gives you a new set of results.

## Runtime differences

The baseline uses a TransformerLens greedy loop with a fixed length and no EOS stopping. HF uses `generate`, and the saved Ollama requests did not fix every decoding option. Chat formatting and defaults may also differ. The HF conversion was checked with a short generated response, not a numerical comparison of model outputs.

The inspected GGUF has `general.file_type = 1`, and the converter requests `f16`. Its label was corrected from Q4_0 to F16. The exact model digest used by the older Ollama requests was not independently verified. These numbers do not measure the effect of 4-bit quantization.

## Reading the comparison

The modified-model runs have fewer keyword refusals than the baseline. Because inputs overlap and settings differ, the gap is not a precise measure of what the weight change alone caused. Use the table as a record of these runs, with the input and runtime notes alongside it.

The CSV, chart, aggregate overlap counts, and source-file hashes are in [results/reference](../results/reference/). Raw prompts and responses are kept local. Model inference was not rerun during repository cleanup.
