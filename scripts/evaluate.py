from deepforest import main
from deepforest import visualize
import matplotlib.pyplot as plt
import os

model = main.deepforest.load_from_checkpoint("outputs/finetuned_treedet.pt")

test_dir = "data/images/test"
test_image = os.listdir(test_dir)[0]
test_path = os.path.join(test_dir, test_image)

predictions = model.predict_image(path=test_path)
visualize.plot_results(predictions)
plt.savefig("outputs/test_prediction.png")
plt.show()
print(predictions)