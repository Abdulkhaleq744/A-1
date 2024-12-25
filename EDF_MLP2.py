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
            self.gradients[self.A] = np.dot(self.outputs[0].gradients[self], self.x.value.T).astype(np.float64)
            self.gradients[self.b] = np.sum(self.outputs[0].gradients[self], axis=1, keepdims=True).astype(np.float64)
            self.gradients[self.x] = np.dot(self.A.value.T, self.outputs[0].gradients[self]).astype(np.float64)
        else:
            print(f"Missing gradients for {self}")


# Sigmoid Activation Node
class Sigmoid(Node):
    def __init__(self, node):
        super().__init__([node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.inputs[0].value).astype(np.float64)

    def backward(self):import numpy as np

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
        print(f"Initialized Parameter: value.shape={self.value.shape}")

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
        # Debugging: Print shapes
        if not isinstance(self.A, Parameter):
            print(f"Error: A is not a Parameter, but {type(self.A)}")
        if not isinstance(self.b, Parameter):
            print(f"Error: b is not a Parameter, but {type(self.b)}")

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
            self.gradients[self.A] = np.dot(self.outputs[0].gradients[self], self.x.value.T).astype(np.float64)
            self.gradients[self.b] = np.sum(self.outputs[0].gradients[self], axis=1, keepdims=True).astype(np.float64)
            self.gradients[self.x] = np.dot(self.A.value.T, self.outputs[0].gradients[self]).astype(np.float64)
        else:
            print(f"Missing gradients for {self}")

# Sigmoid Activation Node
class Sigmoid(Node):
    def __init__(self, node):
        super().__init__([node])
        print(f"Initialized Sigmoid node with input node: {node}")

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.inputs[0].value).astype(np.float64)
        print(f"Sigmoid forward pass: input.shape={self.inputs[0].value.shape}, output.shape={self.value.shape}")

    def backward(self):
        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = (partial * self.outputs[0].gradients[self]).astype(np.float64)
        print(f"Sigmoid backward pass: gradients.shape={self.gradients[self.inputs[0]].shape}")

# Binary Cross Entropy Loss Node
class BCE(Node):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred])
        print(f"Initialized BCE node with y_true: {y_true}, y_pred: {y_pred}")

    def forward(self):
        y_true, y_pred = self.inputs
        self.value = np.sum(-y_true.value * np.log(y_pred.value) - (1 - y_true.value) * np.log(1 - y_pred.value)).astype(np.float64)
        print(f"BCE forward pass: y_true.shape={y_true.value.shape}, y_pred.shape={y_pred.value.shape}, value={self.value}")

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = ((y_pred.value - y_true.value) / (y_pred.value * (1 - y_pred.value))).astype(np.float64)
        self.gradients[y_true] = (np.log(y_pred.value) - np.log(1 - y_pred.value)).astype(np.float64)
        print(f"BCE backward pass: gradients.shape={self.gradients[y_pred].shape}, {self.gradients[y_true].shape}")

        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = (partial * self.outputs[0].gradients[self]).astype(np.float64)


# Binary Cross Entropy Loss Node
class BCE(Node):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        self.value = np.sum(
            -y_true.value * np.log(y_pred.value) - (1 - y_true.value) * np.log(1 - y_pred.value)).astype(np.float64)

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = ((y_pred.value - y_true.value) / (y_pred.value * (1 - y_pred.value))).astype(
            np.float64)
        self.gradients[y_true] = (np.log(y_pred.value) - np.log(1 - y_pred.value)).astype(np.float64)
