"""
Generation utilities — tokenize, generate, hook functions.
References: ortho_cookbook (FailSpy), abliteration article (mlabonne)
"""

import functools
import gc
from typing import List, Callable, Optional

import torch
import einops
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float, Int

from src.config import DEVICE, MAX_NEW_TOKENS, BATCH_SIZE


# ─── Tokenizer functions ─────────────────────────────────────────

def tokenize_instructions_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
) -> Int[torch.Tensor, 'batch_size seq_len']:
    """Tokenize menggunakan apply_chat_template (standard HF)."""
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for instruction in instructions
    ]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids


def tokenize_instructions_fn(
    model: HookedTransformer,
    instructions: List[str],
) -> Int[torch.Tensor, 'batch_size seq_len']:
    """Wrapper untuk tokenize dengan model HookedTransformer."""
    return tokenize_instructions_chat(model.tokenizer, instructions)


def tokenize_instructions_fn_hf(
    tokenizer: AutoTokenizer,
    instructions: List[str],
) -> Int[torch.Tensor, 'batch_size seq_len']:
    """Tokenize untuk model HF."""
    return tokenize_instructions_chat(tokenizer, instructions)


# ─── Generation (TransformerLens) ────────────────────────────────

@torch.no_grad()
def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[torch.Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks: Optional[List] = None,
) -> List[str]:
    """
    Generate dengan hooks untuk inference-time intervention.
    Reference: ortho_cookbook.ipynb (FailSpy)
    """
    all_toks = torch.zeros(
        (toks.shape[0], toks.shape[1] + max_tokens_generated),
        dtype=torch.long,
        device=toks.device,
    )
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks or []):
            logits = model(all_toks[:, :-max_tokens_generated + i])
        next_tokens = logits[:, -1, :].argmax(dim=-1)  # greedy sampling
        all_toks[:, -max_tokens_generated + i] = next_tokens

    return model.tokenizer.batch_decode(
        all_toks[:, toks.shape[1]:], skip_special_tokens=True
    )


@torch.no_grad()
def get_generations(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_fn: Callable,
    fwd_hooks: Optional[List] = None,
    max_tokens_generated: int = MAX_NEW_TOKENS,
    batch_size: int = BATCH_SIZE,
) -> List[str]:
    """
    Generate responses dari model dengan hooks.
    Reference: ortho_cookbook.ipynb
    """
    generations = []
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i + batch_size]
        toks = tokenize_fn(model, batch)
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)
    return generations


# ─── Generation (HF / non-TransformerLens) ──────────────────────

@torch.no_grad()
def get_generations_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = False,
    batch_size: int = BATCH_SIZE,
) -> List[str]:
    """
    Generate menggunakan model HF biasa (lebih ringan).
    Cocok untuk evaluasi setelah model di-ablasi & di-convert.
    """
    generations = []
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i + batch_size]
        tokens = tokenize_instructions_fn_hf(tokenizer, batch)
        input_length = tokens.shape[1]

        out = model.generate(
            tokens.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

        for j in range(len(batch)):
            output = out[j, input_length:]
            text = tokenizer.decode(output, skip_special_tokens=True)
            generations.append(text)

    return generations


# ─── Hook functions (inference-time intervention) ────────────────

def direction_ablation_hook(
    activation: Float[torch.Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[torch.Tensor, "d_act"],
):
    """
    Ablate direction from activation.
    Reference: ortho_cookbook.ipynb (FailSpy)
    Uses einops.einsum for correct projection.
    """
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    proj = einops.einsum(
        activation, direction.view(-1, 1),
        '... d_act, d_act single -> ... single'
    ) * direction
    return activation - proj


def make_ablation_hooks(
    model: HookedTransformer,
    refusal_dir: torch.Tensor,
    layer_indices: Optional[List[int]] = None,
    act_names: Optional[List[str]] = None,
) -> List[tuple]:
    """
    Buat list fwd_hooks untuk ablasi di semua layer tertentu.
    Reference: ortho_cookbook.ipynb
    """
    if layer_indices is None:
        layer_indices = list(range(model.cfg.n_layers))
    if act_names is None:
        act_names = ['resid_pre', 'resid_mid', 'resid_post']

    fwd_hooks = []
    for l in layer_indices:
        for act_name in act_names:
            hook_name = utils.get_act_name(act_name, l)
            hook_fn = functools.partial(direction_ablation_hook, direction=refusal_dir)
            fwd_hooks.append((hook_name, hook_fn))

    return fwd_hooks


# ─── Orthogonalization ──────────────────────────────────────────

def get_orthogonalized_matrix(
    matrix: Float[torch.Tensor, '... d_model'],
    vec: Float[torch.Tensor, 'd_model'],
) -> Float[torch.Tensor, '... d_model']:
    """
    Orthogonalize matrix terhadap direction vector.
    Reference: ortho_cookbook (FailSpy), mlabonne
    Uses einops.einsum for correct multi-dimensional projection.
    """
    proj = einops.einsum(
        matrix, vec.view(-1, 1),
        '... d_model, d_model single -> ... single'
    ) * vec
    return matrix - proj


# ─── Activation extraction helpers ──────────────────────────────

def get_act_idx(cache_dict, act_name: str, layer: int):
    """Helper untuk akses aktivasi dari cache dict."""
    key = (act_name, layer)
    return cache_dict[utils.get_act_name(*key)]


def clear_memory():
    """Flush GPU & CPU cache.
    Panggil setelah operasi berat untuk menghindari OOM."""
    gc.collect()
    torch.cuda.empty_cache()
