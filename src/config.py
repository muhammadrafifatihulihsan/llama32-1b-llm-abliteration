"""
Konfigurasi terpusat — semua path, nama model, hyperparameter ada di sini.
Ubah file ini, bukan hardcode di notebook/script lain.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Root Path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# Load .env dari folder proyek (bukan dari CWD)
load_dotenv(ROOT / ".env")

# ─── Paths ───────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "generations_log"

# Pastikan folder ada
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset ─────────────────────────────────────────────────────
HARMFUL_CSV = DATA_DIR / "harmful_prompts.csv"
HARMLESS_CSV = DATA_DIR / "harmless_prompts.csv"
HOLDOUT_CSV = DATA_DIR / "eval_holdout.csv"

# ─── Model ───────────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
ABLITERATED_DIR = MODELS_DIR / "abliterated"
ABLITERATED_HF_DIR = ABLITERATED_DIR / "hf_format"
GGUF_DIR = MODELS_DIR / "gguf"
GGUF_PATH = GGUF_DIR / "model.gguf"

# ─── Abliteration hyperparameters ───────────────────────────────
N_INST_TRAIN = 256       # Historical extraction count per class
N_INST_TEST = 80         # Historical evaluation count per condition
N_SCREENING = 10         # jumlah kandidat arah yang di-screening
ACTIVATION_LAYERS = ['resid_pre', 'resid_mid', 'resid_post']
SELECTED_LAYERS = ['resid_pre']  # subset layers untuk screening (irit compute)
BATCH_SIZE = 1           # batch size untuk forward pass (4GB VRAM)
BATCH_SIZE_EXTRACT = 4   # batch size untuk ekstraksi aktivasi (lebih kecil)
MAX_NEW_TOKENS = 150
REFUSAL_WEIGHT = 1.0     # strength orthogonalization (1.0=default, >1=stronger)

# ─── Reproducibility ────────────────────────────────────────────
SEED = int(os.getenv("SEED", "42"))

# ─── Device ──────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN", None)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-1b:abliteration")
