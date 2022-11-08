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
from src.layers.convolutional import ConvLayer2D, SuperFastConvLayer2D
from src.layers.dropout import DropoutLayer
from src.models.base_model import SequentialModel
from src.utils.core import convert_categorical2one_hot, convert_prob2categorical
from src.metrics.metrics import softmax_accuracy
from src.optimizers.adam import Adam
from src.utils.plots import lines

from torchvision.datasets import FashionMNIST
from tensorflow.keras.datasets import fashion_mnist

DEBUG = True

if DEBUG:
    # number of samples in the train data set
    N_TRAIN_SAMPLES = 1280
    # number of samples in the test data set
    N_TEST_SAMPLES = 128
    # number of samples in the validation data set
    N_VALID_SAMPLES = 128
    # number of classes
    N_CLASSES = 10
    # image size
    IMAGE_SIZE = 28
else:
    # number of samples in the train data set
    N_TRAIN_SAMPLES = 50000
    # number of samples in the test data set
    N_TEST_SAMPLES = 2500
    # number of samples in the validation data set
    N_VALID_SAMPLES = 250
    # number of classes
    N_CLASSES = 10
    # image size
    IMAGE_SIZE = 28


((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
print("trainX shape:", trainX.shape)
print("trainY shape:", trainY.shape)
print("testX shape:", testX.shape)
print("testY shape:", testY.shape)

X_train = trainX[:N_TRAIN_SAMPLES, :, :]
y_train = trainY[:N_TRAIN_SAMPLES]

X_test = trainX[N_TRAIN_SAMPLES : N_TRAIN_SAMPLES + N_TEST_SAMPLES, :, :]
y_test = trainY[N_TRAIN_SAMPLES : N_TRAIN_SAMPLES + N_TEST_SAMPLES]

X_valid = testX[:N_VALID_SAMPLES, :, :]
y_valid = testY[:N_VALID_SAMPLES]

X_train = X_train / 255
X_train = np.expand_dims(X_train, axis=3)
y_train = convert_categorical2one_hot(y_train)
X_test = X_test / 255
X_test = np.expand_dims(X_test, axis=3)
y_test = convert_categorical2one_hot(y_test)
X_valid = X_valid / 255
X_valid = np.expand_dims(X_valid, axis=3)
y_valid = convert_categorical2one_hot(y_valid)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("X_valid shape:", X_valid.shape)
print("y_valid shape:", y_valid.shape)

# Dropout layer causing issues.

# layers = [
#     # input (N, 28, 28, 1) out (N, 28, 28, 32)
#     SuperFastConvLayer2D.initialize(
#         filters=32, kernel_shape=(3, 3, 1), stride=1, padding="same"
#     ),
#     # input (N, 28, 28, 32) out (N, 28, 28, 32)
#     ReluLayer(),
#     # input (N, 28, 28, 32) out (N, 28, 28, 32)
#     SuperFastConvLayer2D.initialize(
#         filters=32, kernel_shape=(3, 3, 32), stride=1, padding="same"
#     ),
#     # input (N, 28, 28, 32) out (N, 28, 28, 32)
#     ReluLayer(),
#     # input (N, 28, 28, 32) out (N, 14, 14, 32)
#     MaxPoolLayer(pool_size=(2, 2), stride=2),
#     # input (N, 14, 14, 32) out (N, 14, 14, 32)
#     # DropoutLayer(keep_prob=0.9),
#     # input (N, 14, 14, 32) out (N, 14, 14, 64)
#     SuperFastConvLayer2D.initialize(
#         filters=64, kernel_shape=(3, 3, 32), stride=1, padding="same"
#     ),
#     # input (N, 14, 14, 64) out (N, 14, 14, 64)
#     ReluLayer(),
#     # input (N, 14, 14, 64) out (N, 14, 14, 64)
#     SuperFastConvLayer2D.initialize(
#         filters=64, kernel_shape=(3, 3, 64), stride=1, padding="same"
#     ),
#     # input (N, 14, 14, 64) out (N, 14, 14, 64)
#     ReluLayer(),
#     # input (N, 14, 14, 64) out (N, 7, 7, 64)
#     MaxPoolLayer(pool_size=(2, 2), stride=2),
#     # input (N, 7, 7, 64) out (N, 7, 7, 64)
#     # DropoutLayer(keep_prob=0.9),
#     # input (N, 7, 7, 64) out (N, 7 * 7 * 64)
#     FlattenLayer(),
#     # input (N, 7 * 7 * 64) out (N, 256)
#     DenseLayer.initialize(units_prev=7 * 7 * 64, units_curr=256),
#     # input (N, 256) out (N, 256)
#     ReluLayer(),
#     # input (N, 256) out (N, 32)
#     DenseLayer.initialize(units_prev=256, units_curr=32),
#     # input (N, 32) out (N, 32)
#     ReluLayer(),
#     # input (N, 32) out (N, 10)
#     DenseLayer.initialize(units_prev=32, units_curr=N_CLASSES),
#     # input (N, 10) out (N, 10)
#     SoftmaxLayer(),
# ]

layers = [
    # input (N, 28, 28, 1) out (N, 28, 28, 32)
    ConvLayer2D.initialize(
        filters=32, kernel_shape=(3, 3, 1), stride=1, padding="same"
    ),
    # input (N, 28, 28, 32) out (N, 28, 28, 32)
    ReluLayer(),
    # input (N, 28, 28, 32) out (N, 28, 28, 32)
    ConvLayer2D.initialize(
        filters=32, kernel_shape=(3, 3, 32), stride=1, padding="same"
    ),
    # input (N, 28, 28, 32) out (N, 28, 28, 32)
    ReluLayer(),
    # input (N, 28, 28, 32) out (N, 14, 14, 32)
    MaxPoolLayer(pool_size=(2, 2), stride=2),
    # input (N, 14, 14, 32) out (N, 14, 14, 32)
    # DropoutLayer(keep_prob=0.9),
    # input (N, 14, 14, 32) out (N, 14, 14, 64)
    ConvLayer2D.initialize(
        filters=64, kernel_shape=(3, 3, 32), stride=1, padding="same"
    ),
    # input (N, 14, 14, 64) out (N, 14, 14, 64)
    ReluLayer(),
    # input (N, 14, 14, 64) out (N, 14, 14, 64)
    ConvLayer2D.initialize(
        filters=64, kernel_shape=(3, 3, 64), stride=1, padding="same"
    ),
    # input (N, 14, 14, 64) out (N, 14, 14, 64)
    ReluLayer(),
    # input (N, 14, 14, 64) out (N, 7, 7, 64)
    MaxPoolLayer(pool_size=(2, 2), stride=2),
    # input (N, 7, 7, 64) out (N, 7, 7, 64)
    # DropoutLayer(keep_prob=0.9),
    # input (N, 7, 7, 64) out (N, 7 * 7 * 64)
    FlattenLayer(),
    # input (N, 7 * 7 * 64) out (N, 256)
    DenseLayer.initialize(units_prev=7 * 7 * 64, units_curr=256),
    # input (N, 256) out (N, 256)
    ReluLayer(),
    # input (N, 256) out (N, 32)
    DenseLayer.initialize(units_prev=256, units_curr=32),
    # input (N, 32) out (N, 32)
    ReluLayer(),
    # input (N, 32) out (N, 10)
    DenseLayer.initialize(units_prev=32, units_curr=N_CLASSES),
    # input (N, 10) out (N, 10)
    SoftmaxLayer(),
]

if DEBUG:
    # remember to tune down LR if batch size is small.
    optimizer = Adam(lr=0.0001875)

    model = SequentialModel(layers=layers, optimizer=optimizer)

    model.train(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        epochs=10,
        bs=16,
        verbose=True,
    )

else:
    optimizer = Adam(lr=0.003)

    model = SequentialModel(layers=layers, optimizer=optimizer)

    model.train(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        epochs=3,
        bs=256,
        verbose=True,
    )
