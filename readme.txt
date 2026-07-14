Dataset link: https://universe.roboflow.com/models/object-detection

Instructions:
1. Environment setup (conda, in VS Code terminal)

conda create -n treedet python=3.10 -y conda activate treedet pip install deepforest torch torchvision --index-url https://download.pytorch.org/whl/cu118 pip install rasterio geopandas matplotlib

2. Project structure

tree-detection/
??? data/
?   ??? images/          # .tif or .png tiles
?   ??? annotations/     # .csv or .xml boxes
??? scripts/
?   ??? download_data.py
?   ??? train.py
?   ??? evaluate.py
??? outputs/
3. Create scripts/check_cuda.py with:

import torch
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

python scripts/check_cuda.py
4. Try the DeepForest smoke test
from deepforest import main
from deepforest import get_data
from deepforest import visualize
import matplotlib.pyplot as plt

model = main.deepforest()
model.load_model(model_name="weecology/deepforest-tree", revision="main")

sample_image_path = get_data("OSBS_029.png")
predictions = model.predict_image(path=sample_image_path)

visualize.plot_results(predictions)
plt.show()

python scripts/smoke_test.py
5. Download Dataset
On the dataset page, click Download Dataset ? export format: choose "Tensorflow Object Detection CSV"

6. Convert to DeepForest's CSV format

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
        "label": "Tree"
    })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split_name}.csv")
    converted.to_csv(out_path, index=False)
    print(f"Saved {len(converted)} boxes to {out_path}")

if __name__ == "__main__":
    raw_dir = "data/images"       # <-- changed from data/raw
    output_dir = "data/processed"
    for split in ["train", "valid", "test"]:
        convert_split(raw_dir, split, output_dir)	
Run  python scripts/convert_annotations.py
7. Training:

from deepforest import main

model = main.deepforest()
model.load_model(model_name="weecology/deepforest-tree", revision="main")

model.config["gpus"] = 1
model.config["batch_size"] = 2
model.config["train"]["epochs"] = 20          # small dataset, a few more epochs is fine
model.config["train"]["csv_file"] = "data/processed/train.csv"
model.config["train"]["root_dir"] = "data/images/train"

model.config["validation"]["csv_file"] = "data/processed/valid.csv"
model.config["validation"]["root_dir"] = "data/images/valid"

model.create_trainer()
model.trainer.fit(model)

model.save_model("outputs/finetuned_treedet.pt")

Run python scripts/train.py
