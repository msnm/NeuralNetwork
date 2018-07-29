import numpy as np
import numpy.matlib

class NeuralNetwork:


    """ layers is an array containing the layers and the neurons per layer. [2,3,1], has three layers, of 2 inputs
        in layer 1, 3 neurons in the second layer (=hidden layer) and one output in the output layer"""
    def __init__(self, layers):
        self.num_layers = len(layers)  # Size of array represents the number of layers
        self.biases = [np.random.rand(y, 1) for y in layers[1:]]  # Every layer, except the input layer has biases.
        self.weights = [np.random.rand(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def feedforward(self, a):
        activations = [a]
        pre_activations = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a)
            a = self.sigmoid(z)
            activations.append(a)
            pre_activations.append(z)
        return activations, pre_activations

    def train(self, inputs, outputs, epochs=10, learning_rate=0.2, test=False):


        for epoch in range(epochs):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            print("Epoch {}".format(epoch))
            for input, output in zip(inputs, outputs):
                # Algorithm:
                # Step 1: Feedforward the network given the input:                   z_l = w_l * a_l-1 + b_l
                activations, pre_activations = self.feedforward(input.transpose())
                # Step 2: Compute the error vector of the last layer and then backpropate the error for each layer: l-1, l-2, l-3
                error_nabla_b, error_nabla_w = self.backpropagate(activations, pre_activations, output)
                # Step 3: Gradient Descent: Updating the networks weights and biases
                # The gradient of the cost function is the sum of the changes for each w and b
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, error_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, error_nabla_w)]

            self.weights = [w - learning_rate*(nw/len(inputs)) for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - learning_rate*(nb/len(inputs)) for b, nb in zip(self.biases, nabla_b)]

        if test==True:
            for input, output in zip(inputs, outputs):
                activations, pre_activations = self.feedforward(input.transpose())
                print('For input {} was the networks result {} and the actual output {}'.format(input,activations[-1],output))
            self.print_network()


    def backpropagate(self, activations, pre_activations, output):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Error of the final layer, calculated with the quadratic cost function. This error represents the change in C, when delta a changes with a small amount. (partial derivatives)
        error = self.quadric_cost_der(activations[-1], output) * self.sigmoid_der(pre_activations[-1])
        b_error =  error
        w_error = np.dot(error,activations[-2].transpose())
        nabla_b[-1] = b_error
        nabla_w[-1] = w_error

        # Backpropagate the error. If l is the last layer, we start at l-1 .. l = 1. We make use of the negative indices!
        for l in range(2, self.num_layers):
            error = np.dot(self.weights[-l+1].transpose(), error) * self.sigmoid_der(pre_activations[-l])
            b_error = error
            a_error = np.dot(error, activations[-l-1].transpose())
            nabla_b[-l] = b_error
            nabla_w[-l] = a_error

        return nabla_b, nabla_w


    def print_network(self):
        print('The NeuralNetwork contains {} layers'.format(self.num_layers))
        print('The biases of the Network:')
        for c, b in enumerate(self.biases):
         print("==== Layer {} ====".format(c+1))
         print(b)
        print(" ")
        print('The weights of the Network:')
        for c, w in enumerate(self.weights):
         print("==== Layer {} ====".format(c+1))
         print(w)

    def sigmoid(self, z):
     return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_der(self, z):
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))

    """ Derivative of C(a) = 1/2 * (y-a)^2"""
    def quadric_cost_der(self,activation, output):
        return activation - output


if __name__ == '__main__':
    nn = NeuralNetwork([3, 2, 1])
    input = [np.array([[0, 0]]),
             np.array([[1, 0]]),
             np.array([[0, 1]]),
             np.array([[1, 1]])]
    output = np.array([0, 1, 1, 0])

    input = [np.array([[0, 0, 0]]),
             np.array([[0, 0, 1]]),
             np.array([[0, 1, 0]]),
             np.array([[0, 1, 1]]),
             np.array([[1, 0, 0]]),
             np.array([[1, 0, 1]]),
             np.array([[1, 1, 0]]),
             np.array([[1, 1, 1]])]
    output = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    # nn.print_network()
    # result = nn.feedforward(input[0].transpose())
    nn.train(input, output , epochs= 10000, learning_rate=0.5, test=True)
