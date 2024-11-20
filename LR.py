from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define constants
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.02
EPOCHS = 100
TEST_SIZE = 0.25
BATCH_SIZE = 128  # Default batch size for initial training

# Define the means and covariances of the two components
MEAN1 = np.array([0, -1])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([-1, 2])
COV2 = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

# Combine the points and generate labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Plot the generated data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
plt.show()

# Split data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Model parameters
n_features = X_train.shape[1]
n_output = 1

# Initialize weights and biases
A = np.random.randn(n_output, n_features)
b = np.random.randn(n_output)

# Create nodes
x_node = Input()
y_node = Input()
A_node = Parameter(A)
b_node = Parameter(b)

# Build computation graph
linear_node = Linear(A_node, b_node, x_node)
sigmoid = Sigmoid(linear_node)
loss = BCE(y_node, sigmoid)

# Create graph outside the training loop
graph = [x_node, A_node, b_node, linear_node, sigmoid, loss]
trainable = [A_node, b_node]


# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()


def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()


# SGD Update
def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        gradient = t.gradients[t]
        if isinstance(gradient, int):  # Check if gradient is an integer
            gradient = np.array([gradient])
        t.value -= learning_rate * gradient


# Batch processing
def create_batches(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield X[excerpt], y[excerpt]


# Training and evaluating with default BATCH_SIZE
for epoch in range(EPOCHS):
    loss_value = 0
    for X_batch, y_batch in create_batches(X_train, y_train, BATCH_SIZE):
        x_node.value = X_batch
        y_node.value = y_batch.reshape(-1, 1)  # Ensure proper shape

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, LEARNING_RATE)

        loss_value += loss.value

    print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]}")

# Evaluate the model
correct_predictions = 0
for X_batch, y_batch in create_batches(X_test, y_test, BATCH_SIZE):
    x_node.value = X_batch
    forward_pass(graph)
    predictions = (sigmoid.value > 0.5).astype(int)
    correct_predictions += np.sum(predictions == y_batch.reshape(-1, 1))

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = []
for i, j in zip(xx.ravel(), yy.ravel()):
    x_node.value = np.array([i, j]).reshape(1, -1)
    forward_pass(graph)
    Z.append(sigmoid.value)
Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()

# Investigation: Effect of Batch Size on Training Loss
batch_sizes = [2 ** i for i in range(8)]  # Batch sizes 1, 2, 4, 8, 16, 32, 64, 128
batch_loss = []

for batch_size in batch_sizes:
    loss_values = []
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            x_node.value = X_batch
            y_node.value = y_batch.reshape(-1, 1)  # Ensure proper shape

            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, LEARNING_RATE)

            epoch_loss += loss.value

        # Record the average loss for this epoch
        loss_values.append(epoch_loss / X_train.shape[0])

    # Average loss for all epochs for this batch size
    batch_loss.append(np.mean(loss_values))
    print(f"Batch size {batch_size}, Average Loss: {batch_loss[-1]}")

# Plot the training loss for different batch sizes
plt.figure()
plt.plot(batch_sizes, batch_loss, marker='o')
plt.xscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Average Training Loss')
plt.title('Effect of Batch Size on Training Loss')
plt.grid(True)
plt.show()
