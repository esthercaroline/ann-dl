"""A multi-layer perceptron written from scratch with NumPy.

Hidden layers use tanh. The output layer is either a sigmoid (binary, paired
with binary cross-entropy) or a softmax (multi-class, paired with categorical
cross-entropy). Both pairs give the same output gradient, dL/du = (p - y) / N,
so the forward pass, backward pass, and update below are shared verbatim by
every exercise; only the output activation and the loss are arguments.
"""

import numpy as np

EPS = 1e-12


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def softmax(u):
    e = np.exp(u - u.max(axis=1, keepdims=True))  # shift for numerical stability
    return e / e.sum(axis=1, keepdims=True)


def binary_cross_entropy(p, y):
    """L = -mean[y log p + (1 - y) log(1 - p)]; p and y have shape (N, 1)."""
    p = np.clip(p, EPS, 1 - EPS)  # avoid log(0)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def categorical_cross_entropy(p, y):
    """L = -mean over samples of sum_k y_k log p_k; y is one-hot, shape (N, K)."""
    p = np.clip(p, EPS, 1.0)  # avoid log(0)
    return float(-np.mean(np.sum(y * np.log(p), axis=1)))


OUTPUTS = {
    "sigmoid": (sigmoid, binary_cross_entropy),
    "softmax": (softmax, categorical_cross_entropy),
}


class MLP:
    """Fully connected network; ``layer_sizes`` = [n_in, hidden..., n_out]."""

    def __init__(self, layer_sizes, rng, output="sigmoid"):
        self.layer_sizes = list(layer_sizes)
        self.output = output
        self.out_act, self.loss_fn = OUTPUTS[output]
        # Weights ~ N(0, 1/fan_in) keep tanh pre-activations out of saturation.
        self.W = [rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_out))
                  for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros(n_out) for n_out in layer_sizes[1:]]

    def n_params(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    def encode(self, y):
        """Integer labels -> targets shaped like the network output."""
        if self.output == "sigmoid":
            return y.reshape(-1, 1).astype(float)
        return np.eye(self.layer_sizes[-1])[y]

    def forward(self, X):
        """Returns output probabilities and every layer's activation."""
        A = [X]  # A[l] is the input to layer l; backward needs all of them
        for W, b in zip(self.W[:-1], self.b[:-1]):
            A.append(np.tanh(A[-1] @ W + b))  # hidden layer: h = tanh(A W + b)
        P = self.out_act(A[-1] @ self.W[-1] + self.b[-1])  # sigmoid or softmax
        return P, A

    def backward(self, A, P, Y):
        """Gradients of the mean loss w.r.t. every weight and bias."""
        delta = (P - Y) / len(Y)  # dL/du at the output (sigmoid+BCE or softmax+CCE)
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.b)
        for l in reversed(range(len(self.W))):
            grads_W[l] = A[l].T @ delta    # dL/dW = (layer input)^T . (error at its output)
            grads_b[l] = delta.sum(axis=0)  # dL/db = error summed over the batch
            if l > 0:
                # back through W^(l), then through tanh: d tanh = 1 - tanh^2
                delta = (delta @ self.W[l].T) * (1.0 - A[l] ** 2)
        return grads_W, grads_b

    def update(self, grads_W, grads_b, lr):
        """Gradient descent step: theta <- theta - lr * dL/dtheta."""
        for l in range(len(self.W)):
            self.W[l] -= lr * grads_W[l]
            self.b[l] -= lr * grads_b[l]

    def fit(self, X, y, epochs, lr):
        """Full-batch gradient descent; returns the training loss per epoch."""
        Y = self.encode(y)
        losses = []
        for _ in range(epochs):
            P, A = self.forward(X)
            losses.append(self.loss_fn(P, Y))
            grads_W, grads_b = self.backward(A, P, Y)
            self.update(grads_W, grads_b, lr)
        return losses

    def predict(self, X):
        P, _ = self.forward(X)
        if self.output == "sigmoid":
            return (P[:, 0] >= 0.5).astype(int)
        return P.argmax(axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))
