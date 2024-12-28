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

    def SoftMax(self, x):
        # Return SoftMax of layer
        prob = np.exp(x - np.max(x))
        return prob / prob.sum(axis=0)
    
    def ReLU(self, x):
        # Return ReLU of layer
        return np.maximum(x, 0)
    
    def FeedForward(self, x):
        # Set input layer, and initialize list of z values
        self.layers[0] = x
        zs = []

        # Feedforward calclations using linear algebra
        for i in range(self.num_hidden):
            z = np.dot(self.weights[i], self.layers[i]) + self.biases[i]
            zs.append(z)
            self.layers[i+1] = self.ReLU(z)
        z = np.dot(self.weights[-1], self.layers[-2]) + self.biases[-1]
        zs.append(z)
        self.layers[-1] = self.SoftMax(z)

        return self.layers, zs

class Training():
    def __init__(self, FNN, learning_rate = 0.01):
        self.FNN = FNN
        self.learning_rate = learning_rate

    def dReLU(self, x):
        # Return derivative of ReLU function
        return np.where(x > 0, 1, 0)
    
    def gradient_descent(self, inputs, e):
        w_gradients = []
        b_gradients = []

        for i in range(len(inputs)):
            a, z = self.FNN.FeedForward(inputs[i])
            w_gradient, b_gradient = self.backpropogation(a, z, e[i])
            w_gradients.append(w_gradient)
            b_gradients.append(b_gradient)

        w_gradient_avg = [np.mean([w[i] for w in w_gradients], axis=0) for i in range(len(w_gradients[0]))]
        b_gradient_avg = [np.mean([b[i] for b in b_gradients], axis=0) for i in range(len(b_gradients[0]))]

        for i in range(self.FNN.num_hidden + 1):
            self.FNN.weights[i] -= w_gradient_avg[i] * self.learning_rate
            self.FNN.biases[i] -= b_gradient_avg[i] * self.learning_rate

    def backpropogation(self, a, z, e):
        # Initialize gradient vectors
        b_gradient = [np.zeros(b.shape) for b in self.FNN.biases]
        w_gradient = [np.zeros(w.shape) for w in self.FNN.weights]

        # Set output layer gradients, SoftMax with Cross-Entropy Loss
        b_gradient[-1] = a[-1] - e 
        w_gradient[-1] = np.dot(b_gradient[-1], np.transpose(a[-2]))

        # Set hidden layer gradients, ReLU
        for i in range(2, self.FNN.num_hidden + 2):
            b_gradient[-i] = np.dot(np.transpose(self.FNN.weights[-i+1], b_gradient[-i-1])) * self.dReLU(z[-i])
            w_gradient[-i] = np.dot(b_gradient[-i], np.transpose(a[-i-1]))

        # Return gradient vectors
        return w_gradient, b_gradient
    



