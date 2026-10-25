"""
Data loader — load dataset harmful/harmless dari CSV atau langsung dari source.
Semua fungsi return list of strings.
"""

import csv
import random
from typing import List, Tuple
from pathlib import Path

import pandas as pd

from src.config import HARMFUL_CSV, HARMLESS_CSV, HOLDOUT_CSV, SEED

random.seed(SEED)


def load_advbench_harmful(path: Path = None) -> List[str]:
    """
    Load harmful behaviors dari AdvBench.
    Local CSV required; source provenance is described in data/README.md.
    """
    if path is None:
        path = HARMFUL_CSV

    if path.exists():
        df = pd.read_csv(path)
        col = [c for c in df.columns if 'prompt' in c.lower() or 'goal' in c.lower() or 'behavior' in c.lower()]
        if col:
            return df[col[0]].dropna().tolist()
        return df.iloc[:, 0].dropna().tolist()

    raise FileNotFoundError(f"Missing local dataset: {path}. See data/README.md; no automatic substitution is made.")



def load_harmless_alpaca(path: Path = None, n: int = 500) -> List[str]:
    """
    Load harmless instructions dari Alpaca dataset (subset tanpa input).
    Local CSV required; no replacement dataset is downloaded.
    """
    if path is None:
        path = HARMLESS_CSV

    if path.exists():
        df = pd.read_csv(path)
        return df.iloc[:, 0].dropna().tolist()

    raise FileNotFoundError(f"Missing local dataset: {path}. See data/README.md; no automatic substitution is made.")



def create_holdout_set(
    harmful: List[str],
    harmless: List[str],
    n_harmful: int = 100,
    n_harmless: int = 100,
    path: Path = None
) -> Tuple[List[str], List[str]]:
    """
    Ambil subset holdout yang TIDAK dipakai untuk ekstraksi refusal direction.
    Holdout khusus untuk evaluasi final.
    """
    if path is None:
        path = HOLDOUT_CSV

    if path.exists():
        df = pd.read_csv(path)
        harmful_hold = df[df["type"] == "harmful"]["prompt"].tolist()
        harmless_hold = df[df["type"] == "harmless"]["prompt"].tolist()
        return harmful_hold, harmless_hold

    random.shuffle(harmful)
    random.shuffle(harmless)
    harmful_hold = harmful[:n_harmful]
    harmless_hold = harmless[:n_harmless]

    rows = []
    for p in harmful_hold:
        rows.append({"prompt": p, "type": "harmful"})
    for p in harmless_hold:
        rows.append({"prompt": p, "type": "harmless"})
    pd.DataFrame(rows).to_csv(path, index=False)

    return harmful_hold, harmless_hold


def get_harmful_instructions() -> List[str]:
    """Convenience: load all harmful prompts."""
    return load_advbench_harmful()


def get_harmless_instructions() -> List[str]:
    """Convenience: load all harmless prompts."""
    return load_harmless_alpaca()


def get_train_test_split(
    harmful: List[str],
    harmless: List[str],
    holdout_harmful: List[str],
    holdout_harmless: List[str],
    n_train: int = 256,
    n_test: int = 80
):
    """
    Split data: train (untuk ekstraksi refusal direction) + test (untuk evaluasi).
    Holdout tidak dipakai di sini.
    """
    harmful_pool = [p for p in harmful if p not in holdout_harmful]
    harmless_pool = [p for p in harmless if p not in holdout_harmless]

    random.shuffle(harmful_pool)
    random.shuffle(harmless_pool)

    harmful_train = harmful_pool[:n_train]
    harmless_train = harmless_pool[:n_train]
    harmful_test = harmful_pool[n_train:n_train + n_test]
    harmless_test = harmless_pool[n_train:n_train + n_test]

    return harmful_train, harmless_train, harmful_test, harmless_test
