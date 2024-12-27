import numpy as np
import math
import random
import pandas

class FNN():
    def __init__(self):
        self.input_size = 50
        self.num_hidden = 2
        self.hidden_size = 40
        self.output_size = 10

        self.layers = []
        self.weights = []
        self.biases = []

        self.layers.append(np.zeros(self.input_size))
        for i in range(self.num_hidden):
            self.layers.append(np.zeros(self.hidden_size))
        self.layers.append(np.zeros(self.output_size))

        self.weights.append(np.random.randn(self.input_size, self.hidden_size))
        for i in range(self.num_hidden - 1):
            self.weights.append(np.random.randn(self.hidden_size, self.hidden_size))

        for i in range(self.num_hidden):
            self.biases.append(np.zeros(self.hidden_size))
        self.biases.append(np.zeros(self.output_size))

    def SoftMax(x):
        prob = np.exp(x - np.max(x))
        return prob / prob.sum(axis=0)
    
    def ReLU(x):
        return np.maximum(x, 0)
    
    def FeedForward(self):
        for i in range(self.num_hidden + 1):
            buffer = np.dot(self.weights[i], self.layers[i])
            self.layers[i+1] = self.ReLU(buffer + self.biases[i])

        self.layers[-1] = self.SoftMax(self.layers[-1])

        