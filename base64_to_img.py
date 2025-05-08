import io
import os
import base64
import pandas as pd
from pathlib import Path
from PIL import Image

def base64_to_image(base64_string: str) -> Image.Image:
    """Strip header, decode base64 and return a PIL Image."""
    # if your strings include "data:image/…;base64," prefix
    if base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def process_images(
    dataframe: pd.DataFrame,
    number_of_entries: int,
    out_dir: str = "output"
):
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Iterate over the first N rows
    for idx, row in dataframe.iloc[:number_of_entries].iterrows():
        for col in range(3, 9):
            # parse image
            b64 = row[col]              # 5th column
            
            
            

            # split names from 3rd column
            name = str(row[1])
            
            if not isinstance(b64, str) or pd.isna(b64):
                print(f"Row {name}: no Base-64 string -> skipped")
                continue
            
            img = base64_to_image(b64)
            # e.g. output/Alice_0.png
            filename = f"{name}_{col-3}.png"
            dest = os.path.join(out_dir, filename)
            img.save(dest)
            print(f"Saved -> {dest}")

if __name__ == "__main__":
    # load CSV    
    df = pd.read_csv("./emojis.csv")

    # process first 5 rows
    process_images(df, number_of_entries=99, out_dir="images")