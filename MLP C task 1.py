from EDF5 import *
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Create nodes
        self.x_node = Input()
        self.y_node = Input()

        # Initialize weights with better scaling
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        scale3 = np.sqrt(2.0 / output_dim)

        # Network architecture
        self.linear1 = Linear(input_dim, hidden_dim, self.x_node)
        self.linear1.A.value = np.random.randn(hidden_dim, input_dim) * scale1
        self.linear1.b.value = np.zeros((hidden_dim, 1))
        self.relu1 = ReLU(self.linear1)

        self.linear2 = Linear(hidden_dim, output_dim, self.relu1)
        self.linear2.A.value = np.random.randn(output_dim, hidden_dim) * scale2
        self.linear2.b.value = np.zeros((output_dim, 1))

        self.softmax = Softmax(self.linear2)
        self.loss = CrossEntropyLoss(self.y_node, self.softmax)

        # Create graph
        self.graph = [self.x_node, self.y_node,
                      self.linear1, self.relu1,
                      self.linear2, self.softmax, self.loss]
        self.trainable = [self.linear1.A, self.linear1.b,
                          self.linear2.A, self.linear2.b]


    def forward(self, X, training=True):
        """
        Forward pass with shape handling
        Args:
            X: input data (shape: (features, samples))
            training: boolean to indicate if we're training or predicting
        """
        self.x_node.value = X

        # Forward pass through all nodes except loss during prediction
        nodes_to_process = self.graph[:-2] if not training else self.graph
        for node in nodes_to_process:
            node.forward()

        return self.softmax.value

    def backward(self):
        # Initialize gradients for all nodes
        for node in self.graph:
            if isinstance(node, Parameter):
                node.gradients[node] = np.zeros_like(node.value, dtype=np.float64)

        # Perform backward pass
        for node in reversed(self.graph):
            node.backward()

    def train_batch(self, X_batch, y_batch, learning_rate):
        """
        Train on a single batch with proper gradient handling
        """
        # Forward pass
        self.x_node.value = X_batch
        self.y_node.value = y_batch
        self.forward(X_batch, training=True)

        # Backward pass
        self.backward()

        # Update weights with gradient clipping
        for param in self.trainable:
            if param.gradients.get(param) is not None:
                grad = param.gradients[param]

                # Gradient clipping
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1.0:
                    grad = grad / grad_norm

                # Update parameters
                param.value -= learning_rate * grad

    def fit(self, X_train, y_train, epochs, batch_size, learning_rate):
        n_samples = X_train.shape[0]
        best_accuracy = 0
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            num_batches = 0

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]

                # Ensure proper shapes
                X_batch = X_batch.T
                y_batch = y_batch.T

                # Train on batch
                self.train_batch(X_batch, y_batch, learning_rate)

                # Accumulate loss
                epoch_loss += self.loss.value
                num_batches += 1

            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / num_batches

            # Evaluate training accuracy
            if epoch % 1 == 0:  # Check every epoch
                # Get predictions for a batch
                eval_idx = np.random.choice(len(X_train), size=batch_size)
                X_eval = X_train[eval_idx].T
                y_eval = y_train[eval_idx]

                output = self.forward(X_eval, training=False)
                predictions = np.argmax(output, axis=0)
                true_classes = np.argmax(y_eval, axis=1)

                accuracy = np.mean(predictions == true_classes)

                print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}, "
                      f"Training Accuracy: {accuracy:.4f}, LR: {learning_rate:.6f}")

                # Early stopping
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break

    def predict(self, X, batch_size=32):
        """
        Make predictions on input data
        Args:
            X: input data (shape: (n_samples, n_features))
            batch_size: size of batches to process
        Returns:
            predictions: predicted class labels (shape: (n_samples,))
        """
        n_samples = X.shape[0]
        all_predictions = []

        # Process data in batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = X[i:end_idx]
            actual_batch_size = end_idx - i

            # Transpose for network (features, samples)
            batch = batch.T

            # Get predictions for batch
            output = self.forward(batch, training=False)
            batch_predictions = np.argmax(output, axis=0)[:actual_batch_size]
            all_predictions.extend(batch_predictions)

        return np.array(all_predictions)[:n_samples]

    def evaluate(self, X, y, batch_size=32):
        """
        Evaluate model on test data
        Args:
            X: test data (shape: (n_samples, n_features))
            y: true labels (shape: (n_samples,) or (n_samples, n_classes))
            batch_size: size of batches to process
        Returns:
            accuracy: classification accuracy
        """
        predictions = self.predict(X, batch_size=batch_size)

        # Convert one-hot encoded y to class labels if necessary
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)

        # Ensure same number of predictions as labels
        n_samples = len(y)
        predictions = predictions[:n_samples]

        return np.mean(predictions == y)


def main():
    # Data preprocessing
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist['data'], mnist['target'].astype(int)

    # Better data normalization
    X = X.astype(np.float32)
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

    # Model parameters
    input_dim = 784
    hidden_dim = 256
    output_dim = 10
    batch_size = 64
    learning_rate = 0.1  # Higher learning rate
    epochs = 50

    # Initialize and train the model
    print("Training model...")
    model = MLP(input_dim, hidden_dim, output_dim)
    model.fit(X_train, y_train_onehot, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

    # Initialize and train the model
    print("Training model...")
    model = MLP(input_dim, hidden_dim, output_dim)
    model.fit(X_train, y_train_onehot, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    # Evaluate the model
    print("Evaluating model...")
    accuracy = model.evaluate(X_test, y_test_onehot, batch_size=batch_size)
    print(f'Final Test Accuracy: {accuracy * 100:.2f}%')

    # Show sample predictions
    n_samples = 5
    sample_indices = np.random.choice(len(X_test), n_samples)
    X_samples = X_test[sample_indices]
    y_true = y_test[sample_indices]

    predictions = model.predict(X_samples, batch_size=n_samples)

    print("\nSample Predictions:")
    for i in range(n_samples):
        print(f"True: {y_true[i]}, Predicted: {predictions[i]}")


if __name__ == "__main__":
    main()
