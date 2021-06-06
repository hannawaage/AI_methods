# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os
import tqdm

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def dsigmoid(z):
    return sigmoid(z)*(1 - sigmoid(z))

def square_loss(outputs: np.ndarray, targets: np.ndarray):
    diff = targets - outputs
    return 0.5*diff.dot(diff)


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
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        self.inj = []
        self.aj = []

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        self.input_dim = input_dim
        self.hidden_layer = hidden_layer

        # Initialize the weights from random normal distribution
        std = 1/np.sqrt(self.input_dim)
        self.ws = []
        if hidden_layer:
            w_shape = (self.input_dim, self.hidden_units)
            w = np.random.normal(0, std, size=w_shape)
            self.ws.append(w)
            std = 1/np.sqrt(self.hidden_units)
            w_shape = (self.hidden_units, 1)
            w = np.random.normal(0, std, size=w_shape)
            self.ws.append(w)
        else:
            w_shape = (self.input_dim, 1)
            w = np.random.normal(0, std, size=w_shape)
            self.ws.append(w)
        self.grads = [None for i in range(len(self.ws))]

    def reset_output_values(self):
        self.inj = []
        self.aj = []

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

    def bias_trick(self):
        """
        Adds a line to the training/test data to make up the bias to simplify the calculations
        """
        x = self.x_train 
        bias = np.ones((x.shape[0], 1))
        x = np.concatenate((bias, x), axis=1)
        self.x_train= x

        x = self.x_test
        bias = np.ones((x.shape[0], 1))
        x = np.concatenate((bias, x), axis=1)
        self.x_test = x 

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: image of shape 31 (30 + bias)
        Returns:
            a: output of model with shape 1 (probability)
        """
        a = x
        self.aj.append(a)
        for layer in range(len(self.ws)):
            dot_prod = a.T.dot(self.ws[layer])
            self.inj.append(dot_prod)
            a = sigmoid(dot_prod)
            self.aj.append(a)
        return a
    
    def backward(self, output: np.ndarray, target: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            output: output of model
            target: prob of image 
        Updates:
            self.grads: new gradient values for updating the weights
        """

        inj = self.inj[-1]
        ds = dsigmoid(inj)
        delta_j = ds*(target - output)

        # All the "None" steps are because of knoting with numpy
        if self.hidden_layer:
            grad = self.aj[1] * delta_j
            grad = grad[:, None]
            self.grads[1] = grad
            
            in_h = self.inj[0]
            ds = dsigmoid(in_h)
            delta_h = ds*(self.ws[1].dot(delta_j))
            delta_j = delta_h[:, None]
            self.aj[0] = self.aj[0][:, None]
            self.grads[0] = self.aj[0] * delta_j.T
        else:
            grad = self.aj[0] * delta_j
            grad = grad[:, None]
            self.grads[0] = grad

        

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        for _ in tqdm.tqdm(range(self.epochs)):
            for x, y in zip(self.x_train, self.y_train):
                x = np.array(x)
                y = np.array(y)
                outputs = self.forward(x)
                self.backward(outputs, y)
                for layer in range(len(self.ws)):
                    self.ws[layer] = self.ws[layer] + \
                        self.lr*(self.grads[layer])
                self.reset_output_values()
                loss = square_loss(outputs, y)
            
            
        
    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        outputs = self.forward(x)
        return outputs


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
        self.n_features = 31

    
    def get_accuracy(self) -> float:
        #Calculate classification accuracy on the test dataset.
        self.network.load_data()
        self.network.bias_trick()
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
        #Run this method to see if Part 1 is implemented correctly.

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        print('Accuracy w perceptron:', accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')
    
    def test_one_hidden(self) -> None:
        #Run this method to see if Part 2 is implemented correctly.

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        print('Accuracy w hidden layer:', accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')
    

if __name__ == '__main__':
    unittest.main()
