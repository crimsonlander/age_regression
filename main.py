from model_utils import train_model, test_model
from cnn_model_with_output_conv import CNNModel

model = CNNModel()

train_model(model, "cnn_model_3", 20)
#test_model(model, "cnn_model_3")