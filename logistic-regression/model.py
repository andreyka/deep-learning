import numpy as np 

class LogRegression:
    def __init__(self, features_size: int, iterations: int, learning_rate: float):
        self.weights = None
        self.bias = 0.0
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.features_size = features_size

    def reset_parameters(self):
        self.weights = np.random.randn(self.features_size, 1) * 0.01
        self.bias = 0.0

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _forward_propagation(self, W: np.array, b: float, X: np.array) -> np.array:
        Z = np.dot(W.T, X) + b
        A = self._sigmoid(Z)
        return A
    
    def _backward_propagation(self, X: np.array, Y: np.array, A: np.array) -> tuple[np.array, float]:
        m = X.shape[1]
        dZ = A - Y
        dW = np.dot(X, dZ.T) / m
        db = np.sum(dZ) / m
        return dW, db
    
    def _compute_cost(self, A: np.array, Y: np.array) -> float:
        m = Y.shape[1]
        cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return float(cost)
    
    def _update_parameters(self, dW: np.array, db: float):
        self.weights = self.weights - self.learning_rate * dW
        self.bias = self.bias - self.learning_rate * db

    def predict(self, X: np.array) -> np.array:
        return self._forward_propagation(self.weights, self.bias, X)
    
    def fit(self, X_train: np.array, Y_train: np.array):
        costs = []
        self.reset_parameters()
        for i in range(self.iterations):
            A = self._forward_propagation(self.weights, self.bias, X_train)
            L = self._compute_cost(A, Y_train)
            dW, db = self._backward_propagation(X_train, Y_train, A)
            self._update_parameters(dW, db)
            # Record the costs
            if i % 100 == 0:
               costs.append(L)
               print(f"Cost for iteration {i}: {L}")

    def test(self, X_test: np.array, Y_test: np.array):
        A = self.predict(X_test)
        L = self._compute_cost(A, Y_test)
        # Convert probabilities to binary predictions
        predictions = (A > 0.5).astype(int)
        accuracy = 100 - np.mean(np.abs(predictions - Y_test)) * 100
        print(f"Test cost: {L}")
        print(f"Test accuracy: {accuracy:.2f}%")
