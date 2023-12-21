
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

# get data
data = genfromtxt('./advertising/advertising.csv', delimiter=',', skip_header=1)

X = data[:, 0:3]
y = data[:, 3:4]
N = data.shape[0]

# normalize data
maxi = np.max(X)
mini = np.min(X)
avg = np.mean(X)
X = (X-avg) / (maxi-mini)

# get X bar [1,X]
X_b = np.c_[np.ones((N, 1)), X]

def stochastic_gradient_descent():
    epochs_max = 50
    learning_rate = 0.01

    # init theta: [theta0, theta1, theta2, theta3]
    thetas = np.random.randn(4, 1)

    thetas_path = [thetas]
    losses = []
    for epoch in range(epochs_max):
        for i in range(N):
            # get random one sample
            random_index = np.random.randint(N)
            xi = X_b[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]

            # predict y_hat
            y_hat = xi.dot(thetas)

            # compute  loss li
            li = (y_hat - yi) * (y_hat - yi)

            # compute gradient for loss
            d_li = 2 * (y_hat - yi)

            # compute gradient
            gradients = xi.T.dot(d_li)

            # update theta
            thetas = thetas - learning_rate * gradients

            # logging
            thetas_path.append(thetas)
            losses.append(li[0][0])

    return thetas_path, losses


bgd_thetas, losses = stochastic_gradient_descent()

# in loss cho 500 sample đầu
x_axis = list(range(500))
plt.plot(x_axis, losses[:500], color="r")
plt.show()

# print theta
print("theta = ",bgd_thetas[-1])