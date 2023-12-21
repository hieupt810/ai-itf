import csv

import numpy as np
from matplotlib.pylab import plt

n = 9
bias = True
lr = 5e-2


def dataset():
    x = []
    y = []
    neighborhood = {"East": 0, "West": 1, "South": 2, "North": 3}
    with open("house-prices.csv") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader, None)
        for row in reader:
            data = list(map(float, row[2:6]))
            data.append(0 if row[6] == "No" else 1)
            neighbor = [0 for _ in range(4)]
            neighbor[neighborhood[row[7]]] = 1
            data.extend(neighbor)
            data[0] /= 1e3
            x.append(data)
            y.append(float(row[1]) / 1e5)
    return np.array(x), np.array(y)


def loss(predict, groundtruth):
    return (
        1.0
        / (2.0 * groundtruth.shape[0])
        * np.linalg.norm(groundtruth - predict, 2) ** 2
    )


def grad(X, W, Y):
    return 1.0 / X.shape[0] * X.T.dot(X.dot(W.T) - Y)


def train(X, W, Y, X_test, Y_test, epoch=50):
    train_losses = []
    test_losses = []
    for e in range(2, epoch + 1):
        print(f"Epoch {e:5d}:", end=" ")
        predict = X.dot(W.T)
        l = loss(predict, Y)
        train_losses.append(l)
        print(f"Train loss: {l:10.5f}", end="\t")
        g = grad(X, W, Y)
        W = W - (lr * g).T

        test_pred = X_test.dot(W.T)
        test_l = loss(test_pred, Y_test)
        test_losses.append(test_l)
        print(f"Test loss: {test_l:10.5f}")
    return W, train_losses, test_losses


def plot_loss(train_losses, test_losses, epochs=-1, save=True):
    if epochs == -1:
        epochs = len(train_losses)
    epochs = np.arange(2, epochs + 1)
    plt.plot(epochs, train_losses[1:], label="Training loss")
    plt.plot(epochs, test_losses[:-1], label="Test loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    if save:
        plt.savefig("fig.png")
    plt.show()


if __name__ == "__main__":
    # Model
    W = np.random.rand(1, n + 1 if bias else 0)

    # Data
    X, Y = dataset()
    Y = Y.reshape((len(Y), 1))
    if bias:
        X = np.c_[X, np.ones(X.shape[0])]

    train_idx = np.random.choice(np.arange(X.shape[0]), X.shape[0] // 2, replace=False)
    test_idx = np.array(sorted(list(set(range(X.shape[0])) - set(train_idx))))

    train_X = X[train_idx]
    train_Y = Y[train_idx]

    test_X = X[test_idx]
    test_Y = Y[test_idx]

    # Train
    W, train_losses, test_losses = train(train_X, W, train_Y, test_X, test_Y)

    with open("weight.txt", "w") as f:
        f.write(str(W[0]))

    plot_loss(train_losses, test_losses)
