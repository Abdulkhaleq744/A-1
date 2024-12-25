from EDF_MLP3 import *
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
mnist = datasets.load_digits()
X, y = mnist['data'], mnist['target']

# Split the input data into training and testing sets (60% training, 40% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# One-Hot Encoding for the targets
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))


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
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Create nodes
        self.x_node = Input()
        self.y_node = Input()

        self.linear1 = Linear(input_dim, hidden_dim, self.x_node)
        self.relu = ReLU(self.linear1)
        self.linear2 = Linear(hidden_dim, output_dim, self.relu)
        self.softmax = Softmax(self.linear2)

        self.loss = CrossEntropyLoss(self.y_node, self.softmax)

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
            predictions.append(int(self.softmax.value.argmax()))
        return np.array(predictions)


# Initialize and train MLP with automated architecture creation
input_dim = 64  # Each image in MNIST is 8x8 pixels, flattened to a 64-dimensional vector
hidden_dim = 64  # Example hidden layer dimension
output_dim = 10  # There are 10 classes for the digits 0-9
mlp = MLP(input_dim, hidden_dim, output_dim)
mlp.fit(X_train, y_train_onehot, epochs=1000, learning_rate=0.01)

# Predict and evaluate
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'MLP Accuracy on MNIST Dataset: {accuracy * 100:.2f}%')
