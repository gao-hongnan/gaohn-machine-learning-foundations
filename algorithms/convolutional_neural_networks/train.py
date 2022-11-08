import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import sys

sys.path.append(
    r"C:\Users\gaohn\gao_hongnan\gaohn-machine-learning-foundations\algorithms\convolutional_neural_networks"
)

# pylint: disable=import-error, no-name-in-module
from src.activations.relu import ReluLayer
from src.layers.pooling import MaxPoolLayer
from src.activations.softmax import SoftmaxLayer
from src.layers.dense import DenseLayer
from src.layers.flatten import FlattenLayer
from src.layers.convolutional import ConvLayer2D
from src.layers.dropout import DropoutLayer
from src.models.base_model import SequentialModel
from src.utils.core import convert_categorical2one_hot, convert_prob2categorical
from src.metrics.metrics import softmax_accuracy
from src.optimizers.adam import Adam
from src.utils.plots import lines

from torchvision.datasets import FashionMNIST
from tensorflow.keras.datasets import fashion_mnist

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
print("trainX shape:", trainX.shape)
print("trainY shape:", trainY.shape)
print("testX shape:", testX.shape)
print("testY shape:", testY.shape)
