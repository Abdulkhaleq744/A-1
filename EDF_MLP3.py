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
        self.gradients = {self: np.zeros_like(self.value, dtype=np.float64)}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            if self.gradients[self].shape != grad_cost.shape:
                self.gradients[self] = np.zeros_like(grad_cost, dtype=np.float64)
            self.gradients[self] += grad_cost.astype(np.float64)

# Parameter Node
class Parameter(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value.astype(np.float64)

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: np.zeros_like(self.value, dtype=np.float64)}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self].astype(np.float64)

# Linear Node
class Linear(Node):
    def __init__(self, input_dim, output_dim, x):
        self.A = Parameter(np.random.randn(output_dim, input_dim) * np.sqrt(2. / input_dim))
        self.b = Parameter(np.zeros((output_dim, 1)))
        self.x = x
        super().__init__([x])
        print(f"Initialized Linear layer: A.shape={self.A.value.shape}, b.shape={self.b.value.shape}, x={x}")

    def forward(self):
        self.value = np.dot(self.A.value, self.x.value) + self.b.value

    def backward(self):
        self.gradients[self.A] = np.dot(self.outputs[0].gradients[self], self.x.value.T).astype(np.float64)
        self.gradients[self.b] = np.sum(self.outputs[0].gradients[self], axis=1, keepdims=True).astype(np.float64)
        self.gradients[self.x] = np.dot(self.A.value.T, self.outputs[0].gradients[self]).astype(np.float64)

# ReLU Activation Node
class ReLU(Node):
    def __init__(self, node):
        super().__init__([node])

    def forward(self):
        self.value = np.maximum(0, self.inputs[0].value).astype(np.float64)

    def backward(self):
        self.gradients[self.inputs[0]] = (self.outputs[0].gradients[self] * (self.inputs[0].value > 0)).astype(np.float64)

# Softmax Activation Node
class Softmax(Node):
    def __init__(self, node):
        super().__init__([node])

    def forward(self):
        exp_values = np.exp(self.inputs[0].value - np.max(self.inputs[0].value, axis=0))
        probabilities = exp_values / np.sum(exp_values, axis=0)
        self.value = probabilities.astype(np.float64)

    def backward(self):
        # Assuming cross-entropy loss is combined with softmax
        self.gradients[self.inputs[0]] = self.outputs[0].gradients[self].astype(np.float64)

# Cross-Entropy Loss Node
class CrossEntropyLoss(Node):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        self.value = -np.sum(y_true.value * np.log(y_pred.value + 1e-9)).astype(np.float64)  # Adding a small epsilon to avoid log(0)

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = ((y_pred.value - y_true.value) / y_true.value.shape[0]).astype(np.float64)
