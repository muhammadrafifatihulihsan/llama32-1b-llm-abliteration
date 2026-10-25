# Local model artifacts

Weights and generated model files are excluded from this repository. The workflow produced:

- `abliterated/refusal_dirs.safetensors`: extracted direction tensors.
- `abliterated/screening_results.csv`: local screening measurements.
- `abliterated/selected_refusal_dir.safetensors`: selected direction.
- `abliterated/ablation.safetensors`: TransformerLens checkpoint.
- `abliterated/hf_format/`: converted HF weights and tokenizer files.
- `gguf/model.gguf`: GGUF file used by the root `Modelfile`.

The inspected GGUF was F16 (`general.file_type = 1`), not Q4_0. Its existence does not independently verify the model digest used by every historical Ollama request.

The base model is `meta-llama/Llama-3.2-1B-Instruct`. Obtain authorized access separately. No model weights or authorization credentials are included here.
