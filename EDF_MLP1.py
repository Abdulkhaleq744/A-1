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
            self.value = value

    def backward(self):
        self.gradients = {self: np.zeros_like(self.value)}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            if self.gradients[self].shape != grad_cost.shape:
                self.gradients[self] = np.zeros_like(grad_cost)
            self.gradients[self] += grad_cost


# Parameter Node
class Parameter(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: np.zeros_like(self.value)}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


# Linear Node
class Linear(Node):
    def __init__(self, A, b, x):
        self.A = A
        self.b = b
        self.x = x
        super().__init__([x])

    def forward(self):
        # Debugging: Print shapes
        print(f"Forward pass for Linear Node: {self}")
        print(f"Shape of A: {self.A.value.shape}")
        print(f"Shape of x: {self.x.value.shape}")
        print(f"Shape of b: {self.b.value.shape}")

        self.value = np.dot(self.A.value, self.x.value) + self.b.value

    def backward(self):
        # Debugging: Print shapes and gradients
        print(f"Backward pass for Linear Node: {self}")
        if self.outputs:
            print(f"Output gradients: {self.outputs[0].gradients}")

        print(f"Gradients shape check - A: {self.A.value.shape}, x: {self.x.value.shape}")

        if self.outputs and self in self.outputs[0].gradients:
            self.gradients[self.A] = np.dot(self.outputs[0].gradients[self], self.x.value.T)
            self.gradients[self.b] = np.sum(self.outputs[0].gradients[self], axis=1, keepdims=True)
            self.gradients[self.x] = np.dot(self.A.value.T, self.outputs[0].gradients[self])
        else:
            print(f"Missing gradients for {self}")


# Sigmoid Activation Node
class Sigmoid(Node):
    def __init__(self, node):
        super().__init__([node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.inputs[0].value)

    def backward(self):
        partial = self.value * (1 - self.value)
        if self.outputs and self in self.outputs[0].gradients:
            self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]
        else:
            print(f"Missing gradients for {self}")


# Binary Cross Entropy Loss Node
class BCE(Node):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        self.value = np.sum(-y_true.value * np.log(y_pred.value) - (1 - y_true.value) * np.log(1 - y_pred.value))

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (1 / y_true.value.shape[0]) * (y_pred.value - y_true.value) / (
                    y_pred.value * (1 - y_pred.value))
        self.gradients[y_true] = (1 / y_true.value.shape[0]) * (np.log(y_pred.value) - np.log(1 - y_pred.value))
