"""A single-layer perceptron written from scratch (no third-party model).

The activation, prediction, update rule, and training loop are all implemented
here with NumPy only. The same class is reused for both exercises; Exercise 2
turns on the ``pocket`` flag to keep the best-so-far weights on non-separable
data.
"""

import numpy as np


def step(z):
    """Heaviside step: 1 if z >= 0 else 0 (works on scalars or arrays)."""
    return (z >= 0).astype(int)


class Perceptron:
    """Perceptron with the {0, 1} error-driven update rule.

    Prediction:  y_hat = step(w . x + b)
    Update:      w <- w + eta * (y - y_hat) * x
                 b <- b + eta * (y - y_hat)
    """

    def __init__(self, n_features, rng, lr=0.01):
        # Non-zero init: from w = 0 the learning rate only rescales w (see D.3),
        # so we draw small random weights instead.
        self.w = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0
        self.lr = float(lr)

    def predict(self, X):
        """Vectorised prediction for a matrix of samples (n, n_features)."""
        return step(X @ self.w + self.b)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def train(self, X, y, max_epochs=100, pocket=False):
        """Train until a full pass produces no update, or max_epochs.

        Returns a history dict (per-epoch accuracy, best-so-far accuracy, and
        update count) and the pocket record (best weights, bias, accuracy, and
        the epoch at which the best occurred).
        """
        n = len(y)
        best = {
            "w": self.w.copy(),
            "b": self.b,
            "acc": self.accuracy(X, y),
            "epoch": 0,
        }
        history = {"acc": [], "best_acc": [], "updates": []}

        for epoch in range(1, max_epochs + 1):
            updates = 0
            for i in range(n):
                xi, yi = X[i], y[i]
                y_hat = 1 if (self.w @ xi + self.b) >= 0 else 0
                error = yi - y_hat            # 0 if correct, +/-1 on a mistake
                if error != 0:
                    self.w = self.w + self.lr * error * xi
                    self.b = self.b + self.lr * error
                    updates += 1
                    if pocket:
                        acc = self.accuracy(X, y)
                        if acc > best["acc"]:
                            best = {
                                "w": self.w.copy(),
                                "b": self.b,
                                "acc": acc,
                                "epoch": epoch,
                            }

            history["acc"].append(self.accuracy(X, y))
            history["best_acc"].append(best["acc"])
            history["updates"].append(updates)

            if updates == 0:      # a clean pass: the data is separated
                break

        return history, best
