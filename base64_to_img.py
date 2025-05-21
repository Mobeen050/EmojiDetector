import io, os, base64, binascii, logging
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, UnidentifiedImageError


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


def base64_to_image(b64_string: str) -> Optional[Image.Image]:
    """
    Convert a (possibly header-prefixed) base-64 string to a PIL Image.

    Returns None if the string cannot be decoded or is not a valid image.
    """
    if not isinstance(b64_string, str) or not b64_string.strip():
        return None

    # Strip “data:image/…;base64,” if present
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]

    try:
        image_data = base64.b64decode(b64_string, validate=True)
        return Image.open(io.BytesIO(image_data))
    except (binascii.Error, UnidentifiedImageError):
        # Not valid base-64 or not an image
        return None


def process_images(
    dataframe: pd.DataFrame,
    number_of_entries: int,
    out_dir: str = "output"
):
    """Save up to `number_of_entries` rows x 6 columns of base-64 images."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for idx, row in dataframe.iloc[:number_of_entries].iterrows():
        name = str(row[1])              # second column → file prefix

        for col in range(3, 9):         # columns 3 … 8 (0-based)
            img = base64_to_image(row[col])

            if img is None:
                logging.info(f"row {idx} col {col}: skipped — empty/invalid")
                continue

            dest = Path(out_dir) / f"{name}_{col-3}.png"
            try:
                img.save(dest)
                logging.info(f"saved {dest}")
            except Exception as e:      # disk full, permission error, etc.
                logging.warning(f"row {idx} col {col}: could not save ({e})")


if __name__ == "__main__":
    df = pd.read_csv("old_emojis.csv")
    process_images(df, number_of_entries=99, out_dir="images")
