import numpy as np
from ReadMNIST import *
import random
import os

class FNN():
    def __init__(self):
        # Initialize FNN parameters
        self.input_size = 784
        self.num_hidden = 2
        self.hidden_size = 150
        self.output_size = 10
        self.folder_path = os.path.join(os.path.dirname(__file__), "Weights and Biases")

        self.layers = []
        self.weights = []
        self.biases = []

        # Initialize layers, weights, and biases lists
        self.layers.append(np.zeros(self.input_size))
        for i in range(self.num_hidden):
            self.layers.append(np.zeros(self.hidden_size))
        self.layers.append(np.zeros(self.output_size))

        for i in range(self.num_hidden + 1): 
            weight_file = os.path.join(self.folder_path, f'weight_layer_{i}.npy')
            self.weights.append(np.load(weight_file))

        for i in range(self.num_hidden + 1):
            bias_file = os.path.join(self.folder_path, f'bias_layer_{i}.npy')
            self.biases.append(np.load(bias_file))

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
    def __init__(self, learning_rate = 0.01):
        self.FNN = FNN()
        self.learning_rate = learning_rate

    def dReLU(self, x):
        # Return derivative of ReLU function
        return np.where(x > 0, 1, 0)
    
    def gradient_descent(self, inputs, labels):
        # Initialize weight and bias gradients
        w_gradients = []
        b_gradients = []

        for i in range(len(inputs)):
            # Get list of activations, z values, and set up the expected output vector
            a, z = self.FNN.FeedForward(inputs[i])
            expected_output = np.zeros(self.FNN.output_size)
            expected_output[labels[i]] = 1

            # Call backpropogation to get gradient vectors
            w_gradient, b_gradient = self.backpropogation(a, z, expected_output)

            # Add gradient vectors to list for training batch
            w_gradients.append(w_gradient)
            b_gradients.append(b_gradient)

        # Get average gradient vectors
        w_gradient_avg = [np.mean([w[i] for w in w_gradients], axis=0) for i in range(len(w_gradients[0]))]
        b_gradient_avg = [np.mean([b[i] for b in b_gradients], axis=0) for i in range(len(b_gradients[0]))]


        # Adjust weights and biases based off gradient vectors and learning rate
        for i in range(self.FNN.num_hidden + 1):
            self.FNN.weights[i] -= w_gradient_avg[i] * self.learning_rate
            self.FNN.biases[i] -= b_gradient_avg[i] * self.learning_rate

    def backpropogation(self, a, z, e):
        # Initialize gradient vectors
        b_gradient = [np.zeros(b.shape) for b in self.FNN.biases]
        w_gradient = [np.zeros(w.shape) for w in self.FNN.weights]

        # Set output layer gradients, SoftMax with Cross-Entropy Loss
        b_gradient[-1] = a[-1] - e 
        w_gradient[-1] = np.dot(b_gradient[-1].reshape(-1, 1), a[-2].reshape(1, -1))

        # Set hidden layer gradients, ReLU
        for i in range(2, self.FNN.num_hidden + 2):
            b_gradient[-i] = np.dot(self.FNN.weights[-i + 1].T, b_gradient[-i + 1]) * self.dReLU(z[-i])
            w_gradient[-i] = np.dot(b_gradient[-i].reshape(-1, 1), a[-i - 1].reshape(1, -1))

        # Return gradient vectors
        return w_gradient, b_gradient
    
    def update_parameters(self):
        for i, w in enumerate(self.FNN.weights):
            np.save(os.path.join(self.FNN.folder_path, f'weight_layer_{i}.npy'), w)

        for i, b in enumerate(self.FNN.biases):
            np.save(os.path.join(self.FNN.folder_path, f'bias_layer_{i}.npy'), b)
    
    def train(self, num_epochs, batch_size):
        train_images = return_train_images()
        train_labels = return_train_labels()

        train_images = (train_images.reshape(train_images.shape[0], -1)) / 255

        for i in range(num_epochs):
            combined = list(zip(train_images, train_labels))
            random.shuffle(combined)
            shuffled_train_images, shuffled_train_labels = zip(*combined)

            for j in range(1, len(shuffled_train_images) // batch_size):
                index = j * batch_size
                self.gradient_descent(shuffled_train_images[index-batch_size:index], 
                                      shuffled_train_labels[index-batch_size:index])
                percent_done = float(j / (len(shuffled_train_images) // batch_size))
                print(percent_done, end="\r")

            self.learning_rate *= 0.999
            
            self.update_parameters()

            test = Test()
            accuracy = test.evaluate()
            print(f"Accuracy for epoch {str(i)}: {str(accuracy)}")

class Test():
    def __init__(self):
        self.FNN = FNN()
    
    def evaluate(self):
        test_images = return_test_images()
        test_labels = return_test_labels()
        correct = 0

        test_images = (test_images.reshape(test_images.shape[0], -1)) / 255

        for i in range(len(test_images)):
            activations, _ = self.FNN.FeedForward(test_images[i].reshape(-1))
            if np.argmax(activations[-1]) == test_labels[i]: correct += 1

        return correct / len(test_images)

if __name__ == "__main__":
    Train = Training()
    Train.train(50, 60)
