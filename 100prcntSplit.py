# safer_build_dataset_from_raw.py
import random, shutil
from pathlib import Path

RAW_DIR   = Path("images")          # flat list of files like U+1F47D_0.png
DST_ROOT  = Path("emoji_dataset")
TRAIN_DIR = DST_ROOT / "train"
VAL_DIR   = DST_ROOT / "val"
VAL_FRAC  = 0.01
SEED      = 42
random.seed(SEED)

def class_name(fname):            # 'U+1F47D_0.png' -> 'U+1F47D'
    return fname.split("_")[0]

# 1. bucket files by class
buckets = {}
for img in RAW_DIR.glob("*.png"):
    buckets.setdefault(class_name(img.name), []).append(img)

# 2. move/copy with a guarantee: at least 1 stays in train
for cls, files in buckets.items():
    files.sort()                                   # deterministic order
    n_val = max(1, int(len(files) * VAL_FRAC))     # desired val count
    n_val = min(n_val, len(files) - 1)             # leave ≥1 in train

    (TRAIN_DIR/cls).mkdir(parents=True, exist_ok=True)
    (VAL_DIR/cls).mkdir(parents=True,   exist_ok=True)

    val_files = random.sample(files, n_val)
    for img in files:
        dst_dir = VAL_DIR/cls if img in val_files else TRAIN_DIR/cls
        shutil.copy2(img, dst_dir / img.name)
