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

# Multiply Node
class Multiply(Node):
    def __init__(self, x, y):
        super().__init__([x, y])

    def forward(self):
        x, y = self.inputs
        self.value = x.value * y.value

    def backward(self):
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self] * y.value
        self.gradients[y] = self.outputs[0].gradients[self] * x.value

# Addition Node
class Addition(Node):
    def __init__(self, x, y):
        super().__init__([x, y])

    def forward(self):
        x, y = self.inputs
        self.value = x.value + y.value

    def backward(self):
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self]
        self.gradients[y] = self.outputs[0].gradients[self]

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
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]

# Binary Cross Entropy Loss Node
class BCE(Node):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        self.value = np.sum(-y_true.value * np.log(y_pred.value) - (1 - y_true.value) * np.log(1 - y_pred.value))

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (1 / y_true.value.shape[0]) * (y_pred.value - y_true.value) / (y_pred.value * (1 - y_pred.value))
        self.gradients[y_true] = (1 / y_true.value.shape[0]) * (np.log(y_pred.value) - np.log(1 - y_pred.value))

# Linear Node
class Linear(Node):
    def __init__(self, A, b, x):
        self.A = A
        self.b = b
        self.x = x
        super().__init__([x])

    def forward(self):
        self.value = np.dot(self.x.value, self.A.value.T) + self.b.value

    def backward(self):
        self.gradients[self.A] = np.dot(self.outputs[0].gradients[self].T, self.x.value)
        self.gradients[self.b] = np.sum(self.outputs[0].gradients[self], axis=0)
        self.gradients[self.x] = np.dot(self.outputs[0].gradients[self], self.A.value)
