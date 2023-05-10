import modelcompression
from keras.models import load_model

model = load_model("C:\\Users\\HP\\Desktop\\Model Training\\ReluTrained_ReluQuantized\\model_relu.h5")

pruned_model = modelcompression.prune_model(model, 0.8)

