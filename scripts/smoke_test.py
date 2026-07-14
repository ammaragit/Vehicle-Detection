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