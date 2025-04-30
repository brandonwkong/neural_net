Neural Network from Scratch

Description

This project implements a simple feedforward neural network using only NumPy. The network consists of:

Input layer

Two hidden layers

Output layer with sigmoid activation and binary cross-entropy cost

It trains on a toy height/weight dataset and demonstrates forward propagation, cost calculation, and backpropagation.

Installation

Clone the repo or download main.py.

Install dependencies:

pip install numpy

Run training:

python main.py

Project Structure

main.pyCore implementation:

prepare_data() loads and scales toy data

feed_forward() performs forward pass

backprop_layer_*() computes gradients per layer

train() runs epochs and updates weights/biases

Usage

Adjust hyperparameters in train():

alpha (learning rate)

epochs

Prints cost every 20 epochs to track learning.

Next Steps (MNIST Classification)

Load MNIST dataset

from tensorflow.keras.datasets import mnist
(X_train, y_train), _ = mnist.load_data()

Preprocess

Flatten images: X = X_train.reshape(-1, 784).T / 255.0

One-hot encode labels to shape (10, m).

Resize network

n = [784, 128, 64, 10]  # input, two hidden, output

Change activations

Hidden layers: ReLU

Output layer: Softmax with categorical cross-entropy

Backprop updates

Use dZ = A - Y for softmax+CE

Apply ReLU derivative in hidden layers

Evaluation

Compute accuracy: preds = np.argmax(A_out, axis=0)

Plot loss & accuracy curves

Feel free to experiment with more layers, different learning rates, and regularization techniques!
