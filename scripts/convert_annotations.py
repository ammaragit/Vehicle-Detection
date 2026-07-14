import pandas as pd
import os

def convert_split(raw_dir, split_name, output_dir):
    csv_path = os.path.join(raw_dir, split_name, "_annotations.csv")
    df = pd.read_csv(csv_path)

    converted = pd.DataFrame({
        "image_path": df["filename"],
        "xmin": df["xmin"],
        "ymin": df["ymin"],
        "xmax": df["xmax"],
        "ymax": df["ymax"],
        "label": df["class"]   # keep original class names (car/truck/etc) instead of hardcoding
    })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split_name}.csv")
    converted.to_csv(out_path, index=False)
    print(f"Saved {len(converted)} boxes to {out_path}")

if __name__ == "__main__":
    raw_dir = "data/images"
    output_dir = "data/processed"
    for split in ["train", "valid", "test"]:
        convert_split(raw_dir, split, output_dir)