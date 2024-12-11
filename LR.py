from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define constants
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.005  # Adjusted learning rate
EPOCHS = 2000  # Increased epochs
TEST_SIZE = 0.25
BATCH_SIZE = 16  # Adjusted batch size

# Define the means and covariances of the two components
MEAN1 = np.array([1, -1])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([-1, 2])
COV2 = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

# Combine the points and generate labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Manual normalization of data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

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

# Initialize weights and biases with Xavier initialization
A = np.random.randn(n_output, n_features) * np.sqrt(2. / (n_features + n_output))
b = np.zeros((n_output,))

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

# SGD Update with L2 Regularization
def sgd_update(trainables, learning_rate=1e-2, l2_lambda=1e-4):
    for t in trainables:
        gradient = t.gradients[t] + l2_lambda * t.value  # Add L2 regularization term
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
    current_lr = LEARNING_RATE / (1 + epoch / 100)  # Learning rate scheduler
    for X_batch, y_batch in create_batches(X_train, y_train, BATCH_SIZE):
        x_node.value = X_batch
        y_node.value = y_batch.reshape(-1, 1)  # Ensure proper shape

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, current_lr)

        loss_value += loss.value

    print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]}")

    # Early stopping
    if epoch > 50 and loss_value / X_train.shape[0] < 0.01:
        print(f"Early stopping at epoch {epoch + 1}")
        break

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
