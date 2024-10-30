import numpy as np


class NeuralNetwork:
    losses = []
    accuracies = []
    def __init__(self, n_input:int, n_output: int, n_neurons: int, n_layers: int) -> None:
        """
        The construct method help init the weight and biases
        """
        self.cache: list = []
        if n_neurons < 3:
            n_neurons = 3
            
        self.network: dict = {"weights": [], "biases": []}
        self.learning_rate = 0.01

        # Initialize the first hidden layer
        self.network["weights"].append(0.05 * np.random.randn(n_input, n_neurons))
        self.network["biases"].append(np.zeros((1, n_neurons)))

        # Initialize hidden layers
        for _ in range(n_layers - 1):
            self.network["weights"].append(0.05 * np.random.randn(n_neurons, n_neurons))
            self.network["biases"].append(np.zeros((1, n_neurons)))

        # Initialize the output layer
        self.network["weights"].append(0.05 * np.random.randn(n_neurons, n_output))
        self.network["biases"].append(np.zeros((1, n_output)))

    def reLU(self, X) -> np.ndarray:
        return np.maximum(0, X)

    def softmax(self, X) -> np.ndarray:
        normalize = np.exp(X - np.max(X, axis=1, keepdims=True))
        return normalize / np.sum(normalize, axis=1, keepdims=True)

    def reLU_derivative(self, X) -> np.ndarray:
        return X > 0

    def loss(self, y_true, y_pred)-> np.ndarray:
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

    def accuracy(self, y_true, y_pred) -> np.ndarray:
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred_labels == y_true_labels)

    def forward(self, X):
        self.cache = []
        self.linearFunction = X

        for i in range(len(self.network["weights"])):
            temp = np.dot(self.linearFunction, self.network["weights"][i]) + self.network["biases"][i]
            if i < len(self.network["weights"]) - 1:
                self.linearFunction = self.reLU(temp)
            else:
                self.linearFunction = self.softmax(temp)
            self.cache.append((self.linearFunction, self.network["weights"][i], self.network["biases"][i]))

        self.output = self.linearFunction
        return self.output

    def backward(self, X, y_true):
        n_shape = X.shape[0]  
        error = self.output - y_true  

        for i in reversed(range(len(self.network["weights"]))):
            prev_x = self.cache[i - 1][0] if i > 0 else X  

            dW = np.dot(prev_x.T, error) / n_shape
            dB = np.sum(error, axis=0, keepdims=True) / n_shape
            self.network["weights"][i] -= self.learning_rate * dW
            self.network["biases"][i] -= self.learning_rate * dB

            if i > 0:
                error = np.dot(error, self.network["weights"][i].T) * self.reLU_derivative(prev_x)

    def fit(self, X, y, epochs=10000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss(y, y_pred)
            self.losses.append(loss)
            acc = self.accuracy(y, y_pred)
            self.accuracies.append(acc)
            self.backward(X, y)

            # if epoch % 100 == 0:
            #     print(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
