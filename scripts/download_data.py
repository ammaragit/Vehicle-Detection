# scripts/download_data.py
from deepforest import get_data

# Small built-in sample — a few images + CSV annotations
sample_image = get_data("OSBS_029.png")
sample_annotations = get_data("OSBS_029.csv")
print(sample_image, sample_annotations)