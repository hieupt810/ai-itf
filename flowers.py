import csv
import json

import numpy as np


# Activation functions and their gradients
def linear(x):
    return x


def linear_grad(x):
    return np.ones(x.shape)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return np.where(x <= 0, 0, 1)


# Loss functions
def mse(y_hat, y):
    return 1.0 / y.shape[0] * 0.5 * np.linalg.norm(y_hat - y, 2) ** 2, y_hat - y


def binary_cross_entropy(y_hat, y):
    return -np.sum(
        y * np.log(y_hat) + (np.ones(y.shape) - y) * np.log(np.ones(y.shape) - y_hat)
    ), -y / y_hat + (1 - y) / (1 - y_hat)


def categorical_cross_entropy(Y_hat, Y):
    losses = [-np.sum(y * np.log(y_hat)) for y_hat, y in zip(Y_hat, Y)]
    return np.sum(losses), Y_hat - Y


grad_f = {
    linear: linear_grad,
    sigmoid: sigmoid_grad,
    relu: relu_grad,
}


# Optimizer class
class Optimizer:
    container = {}

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, module, grad, hash_id="random"):
        pass

    def container_dict(self):
        container = self.container.copy()
        for key, value in container.items():
            if isinstance(value, np.ndarray):
                container[key] = value.tolist()
        return container

    def load_container(self, container_dict):
        self.container = container_dict
        for key, value in self.container.items():
            try:
                self.container[key] = np.array(value)
            except:
                pass


# Module class
class Module:
    optimizer = Optimizer

    def forward(self, x):
        pass

    def backward(self, loss_grad):
        pass

    def weights(self):
        pass

    def save(self, checkpoint):
        data = {"weights": self.weights(), "optimizer": self.optimizer.container_dict()}
        with open(checkpoint, "w") as f:
            f.write(json.dumps(data))

    def load_weights(self, weights_dict):
        pass

    def load_optimizer(self, optimizer_dict):
        self.optimizer.load_container(optimizer_dict)

    def load(self, checkpoint):
        self.load_weights(checkpoint["weights"])
        self.load_optimizer(checkpoint["optimizer"])


# GD Optimizer
class GD(Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.5):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.container = {}

    def optimize(self, module, grad, hash_id="random"):
        if hash_id not in self.container:
            v_last = np.zeros(module.shape)
        else:
            v_last = self.container.get(hash_id)
        v_now = self.momentum * v_last + self.learning_rate * grad
        module = module - v_now
        self.container[hash_id] = v_now
        return module


# Dense layer
class Dense(Module):
    def __init__(self, in_dim, out_dim, bias=False, activation=linear):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.activation = activation
        self.w = np.random.rand(out_dim, in_dim)
        self.b = None

        if bias:
            self.b = np.random.rand(out_dim, 1)

        self.optimizer = None

    def forward(self, x):
        self.x = x
        if self.bias:
            x = np.c_[x, np.ones(x.shape[0])]
        W = self.w
        if self.bias:
            W = np.c_[W, self.b]
        self.z = W.dot(x.T)
        self.a = self.activation(self.z)
        self.grad_a_z = grad_f[self.activation](self.z)
        return self.a.T

    def backward(self, loss_grad):
        e = np.array(
            [
                grad * a_z_grad if grad.shape == a_z_grad.shape else grad.dot(a_z_grad)
                for grad, a_z_grad in zip(loss_grad, self.grad_a_z.T)
            ]
        )
        e = e.reshape(loss_grad.shape[0], loss_grad.shape[1]).T
        w_grad = e.dot(self.x)
        if self.bias:
            b_grad = np.sum(e, axis=1, keepdims=True)
            self.b = self.optimizer.optimize(self.b, b_grad, str(hash(self)) + "_b")
        self.w = self.optimizer.optimize(self.w, w_grad, str(hash(self)) + "_w")
        return e.T.dot(self.w)

    def weights(self):
        return self.w.tolist()

    def load_weights(self, weights_dict):
        self.w = np.array(weights_dict)


# Softmax layer
class Softmax(Module):
    def __init__(self):
        self.eps = 1e-8

    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        A = e_x / e_x.sum(axis=1, keepdims=True)
        self.A = A
        return A - self.eps

    def backward(self, loss_grad):
        Ds = []
        for id, grad in enumerate(loss_grad):
            Sz = self.A[id]
            D = -np.outer(Sz[:], Sz[:]) + np.diag(Sz.flatten())
            Ds.append(grad * D)
        return np.array(Ds)


# Model class
class Model(Module):
    def __init__(self, layers: list[Module], optimizer):
        self.layers = layers
        self.optimizer = optimizer
        for layer in self.layers:
            layer.optimizer = self.optimizer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)
        return loss_grad

    def weights(self):
        return [layer.weights() for layer in self.layers]

    def load_weights(self, weights_dict):
        for id, weight in enumerate(weights_dict):
            self.layers[id].load_weights(weight)


# Constants and classes
LR = 6e-4
CLASS_MAPPING = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}


# Function to load dataset
def dataset():
    X = []
    Y = []
    with open("input.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            x = list(map(float, row[:4]))
            y = np.zeros(3)
            y[CLASS_MAPPING[row[-1]]] = 1
            X.append(x)
            Y.append(y)
    return np.array(X), np.array(Y)


def test():
    X = []
    with open("output.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            x = list(map(float, row))
            X.append(x)
    return np.array(X)


# Model training and testing
X, Y = dataset()

model = Model(
    [Dense(4, 3, bias=True, activation=sigmoid)], GD(learning_rate=LR, momentum=0.1)
)
for i in range(1000):
    Y_hat = model.forward(X)
    loss, grad = binary_cross_entropy(Y_hat, Y)
    Y_hat = model.forward(X)
    Y_hat_label = np.argmax(Y_hat, axis=1)
    Y_label = np.argmax(Y, axis=1)
    acc = np.sum(Y_hat_label == Y_label)
    print(f"Epoch {i + 1}: loss: {loss}. acc: {acc / X.shape[0]}")
    model.backward(grad)

model.save("weights.txt")

test_X = test()
predicted = model.forward(test_X)
predicted = np.argmax(predicted, axis=1)

for idx in predicted:
    key = list(filter(lambda x: CLASS_MAPPING[x] == idx, CLASS_MAPPING))[0]
    print(key)
