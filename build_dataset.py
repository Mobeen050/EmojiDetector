import os, random, shutil
from pathlib import Path

RAW_DIR   = Path("images")               # where your files are now
DST_ROOT  = Path("emoji_dataset")
TRAIN_DIR = DST_ROOT / "train"
VAL_DIR   = DST_ROOT / "val"
VAL_FRAC  = 0.2                          # 20 % validation split
SEED      = 42

random.seed(SEED)

def class_name(file_name: str) -> str:
    """Extract 'U+1F600' from 'U+1F600_0.png'."""
    return file_name.split("_")[0]

for img_path in RAW_DIR.glob("*.png"):
    cls = class_name(img_path.name)

    # create folders if they do not exist
    (TRAIN_DIR/cls).mkdir(parents=True, exist_ok=True)
    (VAL_DIR/cls).mkdir(parents=True,   exist_ok=True)

    # choose split once per image
    if random.random() < VAL_FRAC:
        dst = VAL_DIR / cls / img_path.name
    else:
        dst = TRAIN_DIR / cls / img_path.name

    shutil.copy2(img_path, dst)          # or .move() if you want to relocate


# from collections import Counter
# import os

# def count_files(path):
#     return Counter(p.parent.name for p in Path(path).rglob("*.png"))

# print("train:", count_files("emoji_dataset/train"))
# print("val  :", count_files("emoji_dataset/val"))
