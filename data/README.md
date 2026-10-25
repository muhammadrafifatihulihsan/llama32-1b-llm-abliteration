# Experiment datasets

The three prompt CSVs used by the local workflow are included in Git and the source ZIP.

| File | Required columns | Recorded local rows |
| --- | --- | ---: |
| `harmful_prompts.csv` | `prompt` (optional `kategori`) | 500 |
| `harmless_prompts.csv` | `prompt` (optional `kategori`) | 500 |
| `eval_holdout.csv` | `prompt,type`; type is `harmful` or `harmless` | 200 |

The holdout contains 100 prompts per class and has no exact-text intersection with either main CSV. It was not used for the published evaluations.

Original source revisions, collection procedures, and transformation history were not fully recorded. Although original code referenced AdvBench/Alpaca downloads, the local CSVs cannot be asserted to be exact copies. No dataset license or redistribution permission is inferred.

Use the included CSVs to execute the model notebooks. The loader raises an error if a primary CSV is absent. Substitute datasets produce a new experiment, not an exact reproduction.
