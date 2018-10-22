#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Oct 2, 2018
@author: You Sheng Lin
"""

# Imports
import numpy as np

# Each row is a training example, each column is a feature  [X1, X2, X3]
X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
y = np.array(([0], [1], [1], [0]), dtype=float)

# Define useful functions
# Activation function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


# Class definition
class NeuralNetwork:

    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)  # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)
        self.layer1 = None
        self.layer2 = None

    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        print(self.layer1)
        print("-" * 10)
        print(self.weights1)
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        print("-" * 10)
        print(self.layer2)
        return self.layer2

    def back_prop(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                 self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feed_forward()
        self.back_prop()


NN = NeuralNetwork(X, y)
for i in range(1):  # trains the NN 1,000 times
    # if i % 1 == 0:
    #     print("for iteration # " + str(i) + "\n")
    #     print("Input : \n" + str(X))
    #     print("Actual Output: \n" + str(y))
    #     print("Predicted Output: \n" + str(NN.feed_forward()))
    #     print("Loss: \n" + str(np.mean(np.square(y - NN.feed_forward()))))  # mean sum squared loss
    #     print("\n")

    NN.train(X, y)