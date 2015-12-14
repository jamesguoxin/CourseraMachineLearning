import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
rng = np.random

def read_file_1v(file_path):
    x = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            x.append(float(line.split(',')[0]))
            y.append(float(line.split(',')[1]))
    return x, y

def read_file_mulv(file_path):
    x = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            values = map(float, line.split(','))
            x.append(values[:-1])
            y.append(values[-1])

    return x, y

def main_1v():
    x, y = read_file_1v("machine-learning-ex1/ex1/ex1data1.txt")
    X = np.asarray(x)
    Y = np.asarray(y)

    x0_value = rng.randn()
    x1_value = rng.randn()

    print x0_value
    print x1_value

    x0 = theano.shared(x0_value, name = 'x0')
    x1 = theano.shared(x1_value, name = 'x1')

    x = T.vector('x')
    y = T.vector('y')

    num_samples = X.shape[0]
    prediction = T.dot(x, x1) + x0
    cost = T.sum(T.pow(prediction-y, 2)) / (2*num_samples)

    gradx0 = T.grad(cost, x0)
    gradx1 = T.grad(cost, x1)

    learning_rate = 0.01
    training_steps = 20000

    train = theano.function([x, y], cost, updates = [(x0, x0 - learning_rate*gradx0), (x1, x1 - learning_rate*gradx1)])
    test = theano.function([x], prediction)

    for i in xrange(training_steps):
        costM = train(X, Y);
        #print costM

    print "x1(slope): " + str(x1.get_value())
    print "x0(intercept): " + str(x0.get_value())

    a = np.linspace(0, 30, 30)
    b = test(a)
    plt.plot(X,Y, 'ro')
    plt.plot(a,b)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

def main_mulv():
    x, y = read_file_mulv("machine-learning-ex1/ex1/ex1data2.txt")
    X = np.asarray(x)
    Y = np.asarray(y)
    # Feature normalization!!
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X = np.subtract(X, np.tile(mu, (X.shape[0], 1)))
    X = np.divide(X, np.tile(sigma, (X.shape[0], 1)))
    #print X

    w_value = rng.randn(X.shape[1], 1)
    b_value = rng.randn()

    print w_value
    print b_value

    w = theano.shared(w_value, name = 'w')
    b = theano.shared(b_value, name = 'b')

    x = T.matrix(name = 'x')
    y = T.vector(name = 'y')

    num_samples = X.shape[0]
    prediction = T.dot(x, w).T + b
    cost = T.sum(T.pow(prediction-y, 2)) / (2*num_samples)

    gradw = T.grad(cost, w)
    gradb = T.grad(cost, b)

    learning_rate = 0.01
    training_steps = 20000

    train = theano.function([x, y], cost, updates = [(w, w - learning_rate*gradw), (b, b - learning_rate*gradb)])
    test = theano.function([x], prediction)

    for i in xrange(training_steps):
        costM = train(X, Y)
        #print costM

    print "w(slope): "
    print w.get_value()
    print "b(intercept): " + str(b.get_value())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    x0 = []
    x1 = []
    for a, b in X:
        x0.append(a)
        x1.append(b)
    ax.scatter(x0, x1, Y)

    a = np.linspace(-3, 5, 20)
    a1 = np.tile(a, (1,1)).T
    b = np.linspace(-3, 5, 20)
    b1 = np.tile(b, (1, 1)).T
    ab = np.concatenate((a1, b1), axis=1)
    c = test(ab)
    c = c.flatten()
    ax.plot(a, b, c, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

if __name__ == "__main__":
    main_1v()
