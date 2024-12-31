# FNN 

# SimpleFNNfromscratch
A simple FeedForward Neural Network made without any frameworks, only using calculus, linear algebra, and NumPy. It consists of a 784 neuron input layer, 2 hidden layers each with 150 neurons, and 10 neuron output layer. I used He initialization for my random initialzaion of weights, and I initialized this biases to just 0. The ReLU activation function was used for the hidden layers and the Softmax activation function was used for the output layer, along with Cross-entropy loss as the cost function. The Neural Network was trained off the MNIST handwritten digits dataset, and was able to achieve a test accuracy of 97.9% after 50 epochs.
