import numpy as np

# Base Node class
class Node:
    def __init__(self, inputs=None):
        self.inputs = inputs if inputs else []
        self.outputs = []
        self.value = None
        self.gradients = {}

        for node in self.inputs:
            node.outputs.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

# Input Node
class Input(Node):
    def forward(self, value=None):
        if value is not None:
            self.value = value.astype(np.float64)

    def backward(self):
        # Initialize gradients with correct shape
        self.gradients = {self: np.zeros_like(self.value, dtype=np.float64)}

        # Sum up gradients from all output nodes
        for n in self.outputs:
            if self not in n.gradients:
                continue

            grad_cost = n.gradients[self]

            # Print shapes for debugging
            # print(f"Gradient shape: {grad_cost.shape}, Expected shape: {self.gradients[self].shape}")

            # Handle different gradient shapes
            if grad_cost.size == self.gradients[self].size:
                # If same number of elements, reshape safely
                grad_cost = grad_cost.reshape(self.gradients[self].shape)
            elif len(grad_cost.shape) == 1:
                # If 1D gradient, broadcast to match input shape
                grad_cost = np.tile(grad_cost[:, np.newaxis], (1, self.gradients[self].shape[1]))
            elif grad_cost.shape[0] == 1:
                # If single sample, broadcast to batch size
                grad_cost = np.tile(grad_cost, (self.gradients[self].shape[0], 1))

            # Ensure the shapes match before adding
            if self.gradients[self].shape == grad_cost.shape:
                self.gradients[self] += grad_cost.astype(np.float64)
            else:
                # If shapes still don't match, create new gradient array
                self.gradients[self] = np.zeros_like(grad_cost, dtype=np.float64)
                self.gradients[self] += grad_cost.astype(np.float64)



class Parameter(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value.astype(np.float64)

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: np.zeros_like(self.value, dtype=np.float64)}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

# Linear Node
# Linear Node
class Linear(Node):
    def __init__(self, A, b, x):
        self.A = A
        self.b = b
        self.x = x
        super().__init__([A, b, x])

    def forward(self):
        # Handle case where b.value is 1D
        b_value = self.b.value.reshape(-1, 1) if len(self.b.value.shape) == 1 else self.b.value
        self.value = np.dot(self.x.value, self.A.value.T) + b_value.T

    def backward(self):
        self.gradients[self.A] = np.dot(self.outputs[0].gradients[self].T, self.x.value)
        self.gradients[self.b] = np.sum(self.outputs[0].gradients[self], axis=0)
        self.gradients[self.x] = np.dot(self.outputs[0].gradients[self], self.A.value)

class Sigmoid(Node):
    def __init__(self, node):
        super().__init__([node])

    def forward(self):
        self.value = 1 / (1 + np.exp(-self.inputs[0].value))

    def backward(self):
        # Ensure proper broadcasting
        grad = self.outputs[0].gradients[self]
        sigmoid_grad = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = grad * sigmoid_grad

class BCE(Node):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        # Add epsilon for numerical stability
        eps = 1e-9
        self.value = -np.mean(
            y_true.value * np.log(y_pred.value + eps) +
            (1 - y_true.value) * np.log(1 - y_pred.value + eps)
        )

    def backward(self):
        y_true, y_pred = self.inputs
        eps = 1e-9
        m = y_true.value.shape[0]
        self.gradients[y_pred] = (-(y_true.value / (y_pred.value + eps) -
                                  (1 - y_true.value) / (1 - y_pred.value + eps))) / m
