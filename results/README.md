# Results

`reference/` contains saved results included with the project:

- `metrics.csv`: three sets of metrics, with ASR explicitly labeled as a heuristic.
- `metrics.png`: chart generated from that CSV.
- `audit.json`: prompt-overlap counts and hashes of the local source files; no raw prompts or responses.

The rates were checked against local generation logs. The old absolute model path was replaced with a readable label, and Q4_0 was corrected to F16 based on the inspected GGUF file.

Read the [results notes](../docs/results.md) for input and runtime differences.

New notebook runs write to `metrics.csv` and `generations_log/` directly under this folder. Those local files are ignored by Git and do not automatically overwrite `reference/`. The plotting script reads the reference CSV.
