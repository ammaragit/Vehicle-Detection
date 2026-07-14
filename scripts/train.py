from deepforest import main

config_args = {
    "num_classes": 4,
    "label_dict": {"car": 0, "2": 1, "hgv": 2, "light": 3}  # fix "2" once you know what it is
}

model = main.deepforest(config_args=config_args)
model.load_model(model_name="weecology/deepforest-tree", revision="main")

model.config["gpus"] = 1
model.config["batch_size"] = 2
model.config["train"]["epochs"] = 10
model.config["train"]["csv_file"] = "data/processed/train.csv"
model.config["train"]["root_dir"] = "data/images/train"

model.config["validation"]["csv_file"] = "data/processed/valid.csv"
model.config["validation"]["root_dir"] = "data/images/valid"

model.create_trainer()
model.trainer.fit(model)

model.save_model("outputs/finetuned_vehicledet.pt")