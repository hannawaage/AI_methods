# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os
import tqdm


class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 3e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        # TODO: Make necessary changes here. For example, assigning the arguments "input_dim" and "hidden_layer" to
        # variables and so forth.

        # Setup for a ANN
        self.input_dim = input_dim
        self.with_hidden_layer = hidden_layer
        self.hidden_layers = []
        self.number_of_hidden_layers = 1
        
        # Enable hidden layers if desirable
        if self.with_hidden_layer:
            self.output = Neuron(self.hidden_units + 1)  # +1 for bias

            # Different amount of perceptron per layer based on position (first have more inputs than the others)
            for i in range(self.number_of_hidden_layers):
                if i == 0:
                    self.hidden_layers.append(Layer(self.input_dim+ 1, self.hidden_units))
                else:
                    self.hidden_layers.append(Layer(self.hidden_units + 1, self.hidden_units))
        else:
            self.output = Neuron(self.input_dim + 1)

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        # TODO: Implement the back-propagation algorithm outlined in Figure 18.24 (page 734) in AIMA 3rd edition.
        # Only parts of the algorithm need to be implemented since we are only going for one hidden layer.

        # Line 6 in Figure 18.24 says "repeat".
        # We are going to repeat self.epochs times as written in the __init()__ method.

        # Line 27 in Figure 18.24 says "return network". Here you do not need to return anything as we are coding
        # the neural network as a class
        for _ in tqdm.tqdm(range(self.epochs)):
            for x,y in zip(self.x_train, self.y_train):

                # Define the input layer
                x = np.array(x)
                y = np.array(y)
                input_layer = InputLayer(x)
                output_from_layer = input_layer.layer_output()
                
                # Calculate the layer outputs
                for layer in self.hidden_layers:
                    output_from_layer = layer.layer_output(output_from_layer) 
                final_output = self.output.activate(self.output.weighted_sum(output_from_layer))

                # Calculate the output gradient:
                output_gradient = sigmoid_derivative(self.output._in_j) * (y - final_output)
                self.output._gradient = output_gradient

                # Calculate the product of the weights and gradient for the output perceptron
                post_gradient_product = sum(self.output.weights*output_gradient)


                # Calculate w_i_j * delta[j] and thus the gradient of all the perceptrons in all of the layers layers.
                for layer_number in reversed(range(len(self.hidden_layers))):
                    for neuron_number in range(len(self.hidden_layers[layer_number].neurons)):
                        
                        # Renaming a variable to improve readability in the if statement below
                        this_neuron = self.hidden_layers[layer_number].neurons[neuron_number]

                        # Check if this is the layer before the output neuron and handle that
                        if layer_number == len(self.hidden_layers) - 1:
                            this_neuron.calculate_gradient(sigmoid_derivative(this_neuron._in_j) * post_gradient_product)
                        else:
                            post_gradient_product =sum(self.hidden_layers[layer_number + 1].get_weights_from_neuron(neuron_number))
                            this_neuron.calculate_gradient(sigmoid_derivative(this_neuron._in_j) * post_gradient_product)

                # Do a gradient decent step for all weights
                for layer in self.hidden_layers:
                    for neuron in layer.neurons:
                        neuron.update_weights(self.lr)
                self.output.update_weights(self.lr)


                


    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        # Define an inputlayer based on the x-values
        input_layer = InputLayer(x)
        
        # Calculate the output of the layers and propagate it through the activation layers of all the layers.
        output_from_layer = input_layer.layer_output()
        for layer in self.hidden_layers:
            output_from_layer = layer.layer_output(output_from_layer) 
        return self.output.activate(self.output.weighted_sum(output_from_layer))

# Return the sigmoid of x
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Return the derivative of the sigmoid of x
def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))


# A class for the input-layer, only saves the x-vector and a bias value
# Used as starting layer for the ANN
class InputLayer:
    def __init__(self, x_values):
        self.bias = -1
        self.x_values = x_values
    # outputs the bias and the x-values
    def layer_output(self):
        return np.append(self.bias, self.x_values)


# A dense layer containing a lot of neurons and a bias node
class Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.bias = -1 
        self.neurons = [Neuron(number_of_inputs) for _ in range(number_of_neurons)]

    # Returns the output of all the perceptrons in the layer
    def layer_output(self, layer_input):
        output = [self.bias]
        for neuron in self.neurons:
            output.append(neuron.activate(neuron.weighted_sum(layer_input)))
        return np.array(output)

    # Calculates the gradient of all the perceptrons in the layers
    def gradient_vector(self):
        gradients = [neuron._gradient for neuron in self.neurons]
        return np.ndarray(gradients)

    # Get the product of the gradient and weights for a given node in all the neuros in the layer (w_i_j)
    def get_weights_from_neuron(self, node_index):
        weights_from_neuron = []
        for node in self.neurons:
            weights_from_neuron.append(node.weights[node_index] * node._gradient)
        return np.array(weights_from_neuron)

# A class for the neurons
class Neuron:
    # Initialize a bunch of variables, most set to None to easily detect bugs
    def __init__(self,number_of_weights):
        self._gradient = None
        self._activation_value = None
        self._in_j = None
        self._input_vector = None
        self.weights = np.array([np.random.randn() for _ in range(number_of_weights)]) 
        
    # The activation function for the neuron, default set to sigmoid
    def activate(self, weighted_sum, activation_function=sigmoid):
        self._activation_value = activation_function(weighted_sum)
        return activation_function(weighted_sum)

    # Calculates a weighted sum of the input values based on the weights
    def weighted_sum(self, input_values):
        self._input_vector = input_values
        self._in_j = self.weights @ input_values
        return self.weights @ input_values

    # Calculates the gradient of a neuron based on w_i_j from the layer to the right (the next layer? Dont know the terminology)
    def calculate_gradient(self, post_layer_gradient_product, activation_function_derivative=sigmoid_derivative):
        self._gradient = activation_function_derivative(self._in_j) * post_layer_gradient_product

    # Update weights based on gradient decent
    def update_weights(self, learning_rate):
        self.weights += learning_rate*self._input_vector*self._gradient



class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        print("Accuracy pereceptron:", accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')
    """
    def test_one_hidden(self) -> None:
        #Run this method to see if Part 2 is implemented correctly.

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        print("Accuracy hidden:", accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')
    """

if __name__ == '__main__':
    unittest.main()
