from typing import Any, Union

import numpy as np
from numpy.core.multiarray import ndarray

""" Network makes a neural network based on a input array , where each index represents a layer and the value
    on an index the number of neurons """


class Network:

    def __init__(self, layers):
        self.nr_layers = len(layers)
        """ Biases starting from the first hidden layer to the output layer. The input layer has no biases """
        self.biases = [np.random.randn(m, 1) for m in layers[1:]]
        """ Weights between input layer and hidden layer 1, ..., weights between hidden layer l-2 and layer l-1,
            weights between layer l-1 and output layer. """
        self.weights = [np.random.randn(n, m) for m, n in zip(layers[:-1], layers[1:])]
        """ Printing the matrices of the neural network """
        self.input = None

    def load_input(self, input_values):
        self.input = np.array(input_values).reshape(len(input_values), 1)
        print(self.input)

    """ activation values of the kth layer, weight and biases of the lth layer """
    def feedforward(self, activation_values_k, weights_l, biases_l):
        z = np.dot(weights_l, activation_values_k) + biases_l
        activation_values_l = sigmoid(z)
        return z, activation_values_l

    def train(self, input, output):
        activation_values = [input]
        z_values = []

        # feedforward
        count = 1
        for w, b in zip(self.weights,self.biases):
            z, a = self.feedforward(activation_values[-1], w,b)
            activation_values.append(a)
            z_values.append(z)
            print("Calculating z {} and a {} for layer {} ".format(z, a, count))
            count+=1

        # backward pass
        # Calculating the partial derivatives for b and w.
        # We start at the output layer, where delta b = 2 * (actual - output). Factor 2 is omitted
        delta_b_output = (activation_values[-1] - output) * sigmoid_der(z_values[-1])
        # delta w = activaiton_k * der_sigmoid _ 2(actual - output)
        delta_w_output = np.dot(activation_values[-2], delta_b_output.transpose())
        print_array(delta_b_output)
        print_array(delta_w_output)

        return activation_values[-1]

    def print_details(self):
        print("Biases")
        print(print_array(self.biases))
        print()
        print("Weights")
        print(print_array(self.weights))
        print()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_der(z):
    return sigmoid(z) * (1-sigmoid(z))


def print_array(array):
    for c, v in enumerate(array):
        print('Layer ' + str(c + 1))
        print(v)
        print()


if __name__ == '__main__':
    net = Network([2, 2, 1])
    net.load_input([1, 0])
    result = net.train(np.array([1,0]),np.array([1]))
    print("Result is {}".format(result))
