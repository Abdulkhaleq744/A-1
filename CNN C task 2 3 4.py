import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define necessary classes

class Conv:
    def __init__(self, num_filters, filter_size, input_channels=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels

        # Initialize filters with shape (num_filters, filter_size, filter_size, input_channels)
        self.filters = np.random.randn(num_filters, filter_size, filter_size, input_channels) / np.sqrt(
            filter_size * filter_size * input_channels)

    def iterate_regions(self, image):
        """
        Generate all possible image regions for convolution
        Args:
            image: input image of shape (height, width, channels)
        """
        h, w, c = image.shape
        new_h = h - self.filter_size + 1
        new_w = w - self.filter_size + 1

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[i:(i + self.filter_size), j:(j + self.filter_size), :]
                yield im_region, i, j

    def forward(self, input):
        """
        Forward pass for conv layer
        Args:
            input: shape (batch_size, height, width, channels)
        """
        self.last_input = input
        batch_size, h, w, channels = input.shape

        # Update input_channels if necessary
        if channels != self.input_channels:
            self.input_channels = channels
            self.filters = np.random.randn(self.num_filters, self.filter_size, self.filter_size,
                                           self.input_channels) / np.sqrt(
                self.filter_size * self.filter_size * self.input_channels)

        new_h = h - self.filter_size + 1
        new_w = w - self.filter_size + 1
        output = np.zeros((batch_size, new_h, new_w, self.num_filters))

        for b in range(batch_size):
            for i in range(new_h):
                for j in range(new_w):
                    im_region = input[b, i:(i + self.filter_size), j:(j + self.filter_size), :]
                    for f in range(self.num_filters):
                        output[b, i, j, f] = np.sum(im_region * self.filters[f])

        return output

    def backprop(self, d_L_d_out, learn_rate):
        """
        Backward pass for conv layer
        Args:
            d_L_d_out: gradient of loss with respect to output
            learn_rate: learning rate for parameter updates
        Returns:
            d_L_d_input: gradient with respect to input
        """
        batch_size = d_L_d_out.shape[0]
        d_L_d_filters = np.zeros_like(self.filters)
        d_L_d_input = np.zeros_like(self.last_input)

        for b in range(batch_size):
            for i in range(d_L_d_out.shape[1]):
                for j in range(d_L_d_out.shape[2]):
                    for f in range(self.num_filters):
                        # Get the corresponding input region
                        im_region = self.last_input[b,
                                    i:i + self.filter_size,
                                    j:j + self.filter_size,
                                    :self.input_channels]
                        # Update gradient for this filter
                        d_L_d_filters[f] += d_L_d_out[b, i, j, f] * im_region

                        # Update gradient for input
                        d_L_d_input[b,
                        i:i + self.filter_size,
                        j:j + self.filter_size,
                        :] += d_L_d_out[b, i, j, f] * self.filters[f]

        # Update filters
        self.filters -= learn_rate * d_L_d_filters / batch_size
        return d_L_d_input


class MaxPooling:
    def __init__(self, filter_size):
        self.filter_size = filter_size

    def iterate_regions(self, image):
        # Handle shape (batch_size, height, width, channels)
        _, h, w, num_filters = image.shape
        new_h = h // self.filter_size
        new_w = w // self.filter_size

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[:,
                            i * self.filter_size:(i * self.filter_size + self.filter_size),
                            j * self.filter_size:(j * self.filter_size + self.filter_size),
                            :]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input

        # Get dimensions
        batch_size, h, w, num_filters = input.shape

        # Calculate output dimensions
        h_out = h // self.filter_size
        w_out = w // self.filter_size

        # Initialize output
        output = np.zeros((batch_size, h_out, w_out, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[:, i, j, :] = np.amax(im_region, axis=(1, 2))

        return output

    def backprop(self, d_L_d_out):
        """
        Backward pass for max pooling layer
        Args:
            d_L_d_out: gradient of loss with respect to output
        Returns:
            d_L_d_input: gradient with respect to input
        """
        d_L_d_input = np.zeros_like(self.last_input)

        for b in range(d_L_d_out.shape[0]):
            for i in range(d_L_d_out.shape[1]):
                for j in range(d_L_d_out.shape[2]):
                    for c in range(d_L_d_out.shape[3]):
                        # Get the corresponding input region
                        i_start = i * self.filter_size
                        j_start = j * self.filter_size
                        i_end = i_start + self.filter_size
                        j_end = j_start + self.filter_size

                        input_region = self.last_input[b, i_start:i_end, j_start:j_end, c]

                        # Find location of maximum
                        mask = input_region == np.max(input_region)

                        # Distribute gradient
                        d_L_d_input[b, i_start:i_end, j_start:j_end, c] += \
                            d_L_d_out[b, i, j, c] * mask

        return d_L_d_input


class ReLU:
    def __init__(self, node):
        self.node = node

    def forward(self, x):
        self.last_input = x
        return np.maximum(0, x)

    def backward(self, d_L_d_out):
        return d_L_d_out * (self.last_input > 0)

class Linear:
    def __init__(self, input_len, output_len):
        self.weights = np.random.randn(input_len, output_len) * np.sqrt(2. / input_len)
        self.biases = np.zeros(output_len)

    def forward(self, x):
        self.last_input_shape = x.shape
        self.last_input = x
        self.last_output = np.dot(x, self.weights) + self.biases
        return self.last_output

    def backward(self, d_L_d_out, learn_rate):

        # Compute gradients
        d_L_d_input = np.dot(d_L_d_out, self.weights.T)
        d_L_d_weights = np.dot(self.last_input.T, d_L_d_out)
        d_L_d_biases = np.sum(d_L_d_out, axis=0)

        # Update parameters
        self.weights -= learn_rate * d_L_d_weights
        self.biases -= learn_rate * d_L_d_biases

        return d_L_d_input

class Softmax:
    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.probs

    def backward(self, d_L_d_out):
        return d_L_d_out


class CNN:
    def __init__(self):
        # Initialize with proper input channels
        self.conv1 = Conv(16, 3, input_channels=1)  # First layer expects 1 channel
        self.pool1 = MaxPooling(2)
        self.conv2 = Conv(32, 3, input_channels=16)  # Second layer expects 16 channels
        self.pool2 = MaxPooling(2)
        self.conv3 = Conv(64, 3, input_channels=32)  # Third layer expects 32 channels
        self.pool3 = MaxPooling(2)
        self.linear = Linear(64, 10)  # Final layer

    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: input data (batch_size, height, width, channels)
        """
        # Store intermediate shapes for backprop
        self.layer_outputs = []

        # Conv1 + Pool1
        x = self.conv1.forward(x)
        self.layer_outputs.append(x)
        x = self.pool1.forward(x)
        self.layer_outputs.append(x)

        # Conv2 + Pool2
        x = self.conv2.forward(x)
        self.layer_outputs.append(x)
        x = self.pool2.forward(x)
        self.layer_outputs.append(x)

        # Conv3 + Pool3
        x = self.conv3.forward(x)
        self.layer_outputs.append(x)
        x = self.pool3.forward(x)

        # Store the shape before flattening
        self.pre_flatten_shape = x.shape

        # Flatten and Linear
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.linear.forward(x)
        return x

    def backprop(self, d_L_d_out, learn_rate):
        """
        Backward pass through the network
        Args:
            d_L_d_out: gradient of loss with respect to output
            learn_rate: learning rate for parameter updates
        """
        # Backprop through linear layer
        d_L_d_out = self.linear.backward(d_L_d_out, learn_rate)

        # Reshape back to match the last convolutional layer output
        d_L_d_out = d_L_d_out.reshape(self.pre_flatten_shape)

        # Backprop through Conv3 + Pool3
        d_L_d_out = self.pool3.backprop(d_L_d_out)
        if d_L_d_out is None:
            return
        d_L_d_out = self.conv3.backprop(d_L_d_out, learn_rate)

        # Backprop through Conv2 + Pool2
        d_L_d_out = self.pool2.backprop(d_L_d_out)
        if d_L_d_out is None:
            return
        d_L_d_out = self.conv2.backprop(d_L_d_out, learn_rate)

        # Backprop through Conv1 + Pool1
        d_L_d_out = self.pool1.backprop(d_L_d_out)
        if d_L_d_out is None:
            return
        d_L_d_out = self.conv1.backprop(d_L_d_out, learn_rate)

        return d_L_d_out


def cross_entropy_loss(predictions, targets):
    """
    Compute cross entropy loss
    Args:
        predictions: model output (batch_size, num_classes)
        targets: one-hot encoded labels (batch_size, num_classes)
    Returns:
        loss: scalar value
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15

    # Clip predictions to avoid numerical instability
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    # Compute softmax
    exp_values = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # Compute cross entropy loss
    batch_size = targets.shape[0]
    correct_confidences = np.sum(probabilities * targets, axis=1)
    loss = -np.log(correct_confidences + epsilon)

    return np.mean(loss)


def preprocess_data(X, image_size=28):
    """
    Preprocess the data to ensure proper dimensions
    """
    # Reshape and add channel dimension
    X = X.reshape(-1, image_size, image_size, 1)
    # Normalize to [0, 1]
    X = X / 255.0
    return X


# Update create_batches function
def create_batches(X, y, batch_size):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, n_samples - batch_size + 1, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        yield X_batch, y_batch


# Update the main training code
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X, y = mnist['data'], mnist['target'].astype(int)

# Preprocess data
X = preprocess_data(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-Hot Encoding for the targets
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

# Training parameters
batch_size = 32
epochs = 10
learning_rate = 0.01

# Initialize CNN
cnn = CNN()

# Training loop
print("Training model...")
for epoch in range(epochs):
    total_loss = 0
    n_batches = 0

    for X_batch, y_batch in create_batches(X_train, y_train_onehot, batch_size=batch_size):
        # Forward pass
        output = cnn.forward(X_batch)

        # Compute loss
        loss = cross_entropy_loss(output, y_batch)
        total_loss += loss
        n_batches += 1

        # Compute gradients
        exp_values = np.exp(output - np.max(output, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        d_L_d_out = probabilities - y_batch

        # Backward pass
        cnn.backprop(d_L_d_out, learning_rate)

        if n_batches % 10 == 0:
            print(f'Epoch {epoch + 1}, Batch {n_batches}, Loss: {loss:.4f}')

    # Compute average loss for the epoch
    avg_loss = total_loss / n_batches

    # Compute training accuracy on a subset
    subset_size = min(1000, len(X_train))
    indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_subset = X_train[indices]
    y_subset = y_train_onehot[indices]

    train_predictions = cnn.forward(X_subset)
    train_acc = accuracy_score(np.argmax(y_subset, axis=1),
                               np.argmax(train_predictions, axis=1))

    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Training Accuracy: {train_acc:.4f}')
