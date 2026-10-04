import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MOSAIC_DIR", HERE.parent.parent))
ROSEF = Path(os.environ.get("ROSEF_ROOT", MODEL_DIR.parent))
MOSAIC_CODE = MODEL_DIR / "code"
MOSAIC_WEIGHTS = MODEL_DIR / "weights" / "rag_mosaic_best_ep12.pth"
TEXT_ASSETS = MODEL_DIR / "weights" / "text_assets_qwen3emb8b.pt"
WORK = Path(os.environ.get("BM_WORK", ROSEF / "brainmosaic" / "work"))
CHANLOCS = WORK / "chanlocs_105.json"
REPLAY_DIR = WORK / "eeg" / "zuco1_SR"
REPLAY_SUBJECTS = ["ZAB", "ZJM", "ZKW"]

QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-1.7B")
LLM_DEVICE = os.environ.get("LLM_DEVICE", "mps")
DECODER_DEVICE = os.environ.get("DECODER_DEVICE", "cpu")

FS = 250
N_CHANNELS = 105
EPOCH_SECONDS = 9
BUFFER_SECONDS = 120
PORT = int(os.environ.get("PORT", 5050))
