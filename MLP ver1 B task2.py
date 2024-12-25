from EDF_MLP1 import *
import numpy as np

# Generate XOR dataset
np.random.seed(0)
N = 100
X = np.random.randn(N, 2)
y = np.array((X[:, 0] > 0) ^ (X[:, 1] > 0), dtype=int)


# Define the MLP architecture
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights and biases
        self.W1 = Parameter(np.random.randn(hidden_dim, input_dim) * np.sqrt(2. / input_dim))
        self.b1 = Parameter(np.zeros((hidden_dim, 1)))
        self.W2 = Parameter(np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2. / hidden_dim))
        self.b2 = Parameter(np.zeros((hidden_dim, 1)))
        self.W3 = Parameter(np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2. / hidden_dim))
        self.b3 = Parameter(np.zeros((hidden_dim, 1)))
        self.W4 = Parameter(np.random.randn(output_dim, hidden_dim) * np.sqrt(2. / hidden_dim))
        self.b4 = Parameter(np.zeros((output_dim, 1)))

        # Create nodes
        self.x_node = Input()
        self.y_node = Input()
        self.linear1 = Linear(self.W1, self.b1, self.x_node)
        self.sigmoid1 = Sigmoid(self.linear1)
        self.linear2 = Linear(self.W2, self.b2, self.sigmoid1)
        self.sigmoid2 = Sigmoid(self.linear2)
        self.linear3 = Linear(self.W3, self.b3, self.sigmoid2)
        self.sigmoid3 = Sigmoid(self.linear3)
        self.linear4 = Linear(self.W4, self.b4, self.sigmoid3)
        self.sigmoid4 = Sigmoid(self.linear4)

        self.loss = BCE(self.y_node, self.sigmoid4)

    def forward(self):
        for layer in [self.linear1, self.sigmoid1, self.linear2, self.sigmoid2, self.linear3, self.sigmoid3,
                      self.linear4, self.sigmoid4, self.loss]:
            layer.forward()

    def backward(self):
        for layer in [self.loss, self.sigmoid4, self.linear4, self.sigmoid3, self.linear3, self.sigmoid2, self.linear2,
                      self.sigmoid1, self.linear1][::-1]:
            layer.backward()

    def update_weights(self, learning_rate):
        for param in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]:
            # Debugging: Print parameter gradients
            if param in param.gradients:
                print(f"Updating weight {param} with gradients {param.gradients[param]}")
                param.value -= learning_rate * param.gradients[param]
            else:
                print(f"Missing gradients for parameter {param}")

    def fit(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            epoch_loss = 0
            for x, y in zip(X_train, y_train):
                self.x_node.value = x.reshape(-1, 1)
                self.y_node.value = y.reshape(-1, 1)

                self.forward()

                # Debugging: Print loss value
                print(f"Loss value: {self.loss.value}")

                epoch_loss += self.loss.value
                self.backward()
                self.update_weights(learning_rate)

            epoch_loss /= len(X_train)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss}")

    def predict(self, X):
        predictions = []
        for x in X:
            self.x_node.value = x.reshape(-1, 1)
            self.forward()
            predictions.append(int(self.sigmoid4.value > 0.5))
        return np.array(predictions)


# Initialize and train MLP
input_dim = 2
hidden_dim = 20
output_dim = 1
mlp = MLP(input_dim, hidden_dim, output_dim)
mlp.fit(X, y, epochs=5000, learning_rate=0.001)

# Predict and evaluate
y_pred = mlp.predict(X)
accuracy = np.mean(y_pred == y)
print(f'MLP Accuracy on XOR Dataset: {accuracy * 100:.2f}%')
