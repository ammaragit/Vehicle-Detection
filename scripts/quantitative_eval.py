from deepforest import main

model = main.deepforest.load_from_checkpoint("outputs/finetuned_vehicledet.pt")

results = model.evaluate(
    csv_file="data/processed/test.csv",
    root_dir="data/images/test",
    iou_threshold=0.4
)

print("Box precision:", results["box_precision"])
print("Box recall:", results["box_recall"])