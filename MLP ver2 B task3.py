from EDF_MLP2 import *
import numpy as np

# Generate XOR dataset
np.random.seed(0)
N = 100
X = np.random.randn(N, 2)
y = np.array((X[:, 0] > 0) ^ (X[:, 1] > 0), dtype=int)


# Helper function for topological sort
def topological_sort(feed_dict):
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outputs:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outputs:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m)

    return L


# Define the MLP architecture
class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Create nodes dynamically based on hidden_dims
        self.x_node = Input()
        self.y_node = Input()

        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = []

        for i in range(len(dims) - 1):
            layer = Linear(dims[i], dims[i + 1], self.x_node if i == 0 else self.layers[-1])
            self.layers.append(layer)
            self.layers.append(Sigmoid(layer))

        self.loss = BCE(self.y_node, self.layers[-1])

        # Create graph and trainable lists
        self.graph = topological_sort({self.x_node: None, self.y_node: None})
        self.trainable = [n for n in self.graph if isinstance(n, Parameter)]

    def forward(self):
        for layer in self.graph:
            layer.forward()

    def backward(self):
        for layer in reversed(self.graph):
            layer.backward()

    def update_weights(self, learning_rate):
        for param in self.trainable:
            param.value -= learning_rate * param.gradients[param]

    def fit(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            epoch_loss = 0
            for x, y in zip(X_train, y_train):
                self.x_node.value = x.reshape(-1, 1).astype(np.float64)
                self.y_node.value = y.reshape(-1, 1).astype(np.float64)

                self.forward()
                epoch_loss += self.loss.value
                self.backward()
                self.update_weights(learning_rate)

            epoch_loss /= len(X_train)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss}")

    def predict(self, X):
        predictions = []
        for x in X:
            self.x_node.value = x.reshape(-1, 1).astype(np.float64)
            self.forward()
            predictions.append(int(self.layers[-1].value > 0.5))
        return np.array(predictions)


# Initialize and train MLP with automated architecture creation
input_dim = 2
hidden_dims = [20, 20, 20]  # Example hidden layer dimensions
output_dim = 1
mlp = MLP(input_dim, hidden_dims, output_dim)
mlp.fit(X, y, epochs=5000, learning_rate=0.001)

# Predict and evaluate
y_pred = mlp.predict(X)
accuracy = np.mean(y_pred == y)
print(f'MLP Accuracy on XOR Dataset: {accuracy * 100:.2f}%')
