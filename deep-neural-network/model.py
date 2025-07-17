import numpy as np

from typing import List

class DeepNeuralNetwork:
    def __init__(self, dimensions: List[int], activations: List[str], iterations: int, learning_rate:float):
        self.activations = activations
        self.dimensions = dimensions
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.parameters = {}
        self.cache = {}
        self.grads = {}
    
    # Sigmoid activation function and its derivative
    def _sigmoid(self, z, derivative=False):
        if derivative:
            sig = self._sigmoid(z)
            return sig * (1 - sig)
        
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))
    
    # ReLU activation function and its derivative
    def _relu(self, z, derivative=False):
        if derivative:
            return (z > 0).astype(float)
        
        return np.maximum(0, z)
    
    # Tanh activation function and its derivative
    def _tanh(self, z, derivative=False):
        if derivative:
            return 1 - self._tanh(z) ** 2
        
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    
    # Leaky ReLU activation function and its derivative
    def _leaky_relu(self, z, alpha=0.01, derivative=False):
        if derivative: 
             return np.where(z > 0, 1, alpha)
        
        return np.maximum(z, alpha * z)
    
    # Softmax activation function and its derivative
    def _softmax(self, z, derivative=False):
        if derivative:
            s = self._softmax(z)
            return s * (1 - s)
    
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # Selects acitvation function for a given layer
    def _get_activation_for_layer(self, activation: str) -> callable:
        if activation == 'sigmoid':
            return self._sigmoid

        elif activation == 'relu':
            return self._relu
        
        elif activation == 'leaky_relu':
            return self._leaky_relu
        
        elif activation == 'tanh':
            return self._tanh

        else:
            return self._softmax            

    def _forward_propagation_layer(self, A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, layer:int) -> tuple[np.ndarray, np.ndarray]:
        Z = np.dot(W, A_prev) + b
        activation = self._get_activation_for_layer(self.activations[layer - 1])
        A = activation(Z)
        return A, Z
    
    def _forward_propagation_network(self, X:np.ndarray):
        A_prev = X
        self.cache['A0'] = X
        for i in range(len(self.dimensions) -1):
            W = self.parameters['W' + str(i+1)]
            b = self.parameters['b' + str(i+1)]
            A, Z = self._forward_propagation_layer(A_prev, W, b, i + 1)
            # Saving values to cache for backward propagation
            self.cache['A' + str(i+1)] = A
            self.cache['Z' + str(i+1)] = Z
            self.cache['W' + str(i+1)] = W
            self.cache['b' + str(i+1)] = b
            A_prev = A

        return A

    def _backward_propagation_layer(self, dA: np.ndarray, layer:int):
        A_prev = self.cache['A' + str(layer - 1)]
        m = A_prev.shape[1]
        Z = self.cache['Z' + str(layer)]
        W = self.cache['W' + str(layer)]
        activation_name = self.activations[layer - 1]
        # Calculate the derivative of the activation function
        if activation_name == 'softmax':
             dZ = dA
        else:
             activation = self._get_activation_for_layer(activation_name)
             dZ = dA * activation(Z, derivative=True)

        # Calculate gradients
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def _backward_propagation_network(self, A:np.array, Y: np.array):
        if self.activations[-1] == 'softmax':
            dA = A - Y  # For softmax + cross-entropy
        else:
            dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
        
        for i in reversed(range(1, len(self.dimensions))): 
            dA_prev, dW, db = self._backward_propagation_layer(dA, i)
            # Saving values to cache for backward propagation
            self.cache['dA' + str(i-1)] = dA_prev
            self.grads['dW' + str(i)] = dW
            self.grads['db' + str(i)] = db
            dA = dA_prev  # Update dA for the next layer

    def _update_parameters(self):
        for i in range(len(self.dimensions) -1):
            self.parameters['W' + str(i+1)] -= self.learning_rate * self.grads['dW' + str(i+1)] 
            ## Biases can be initialized by zeroes
            self.parameters['b' + str(i+1)] -= self.learning_rate * self.grads['db' + str(i+1)] 

    def _compute_cost(self, A: np.ndarray, Y: np.ndarray) -> float:
        m = Y.shape[1]
        # Categorical cross-entropy loss
        A_clipped = np.clip(A, 1e-15, 1 - 1e-15)
        cost = -1 / m * np.sum(Y * np.log(A_clipped) + (1 - Y) * np.log(1 - A_clipped))
        return np.squeeze(cost)

    def initialize_parameters(self):
        # We set parameters only starting from layer 1 as layer 0 is features.
        for i in range(len(self.dimensions) - 1):
            layer_size = self.dimensions[i + 1]
            prev_layer_size = self.dimensions[i]
            # Weights should be initialized by random values
            self.parameters['W' + str(i+1)] = np.random.randn(layer_size, prev_layer_size) * np.sqrt(2.0 / prev_layer_size)
            ## Biases can be initialized by zeroes
            self.parameters['b' + str(i+1)] = np.zeros((layer_size, 1))

    def fit(self, X_train:np.ndarray, Y_train:np.ndarray):
        self.initialize_parameters()

        for i in range(self.iterations):
            # Forward propagation round
            A = self._forward_propagation_network(X_train)
            # Checking the current cost function value
            cost = self._compute_cost(A, Y_train)
            # Backward propagation and updates of parameters
            self._backward_propagation_network(A, Y_train)
            self._update_parameters()

            # Print cost after every 100 iterations
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X_test):
        A = self._forward_propagation_network(X_test)
        # If output layer has more than one neuron, assume multiclass
        if A.shape[0] > 1:
            # Return one-hot encoded predictions
            predictions = np.zeros_like(A)
            predictions[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
            return predictions
        # If output layer has one neuron, assume binary classification
        else:
            return (A > 0.5).astype(int)
    
    def test(self, X_test: np.array, Y_test: np.array):
        A = self.predict(X_test)
        L = self._compute_cost(A, Y_test)
        # Convert probabilities to binary predictions
        predictions = (A > 0.5).astype(int)
        accuracy = 100 - np.mean(np.abs(predictions - Y_test)) * 100
        print(f"Test cost: {L}")
        print(f"Test accuracy: {accuracy:.2f}%")
        return accuracy
    