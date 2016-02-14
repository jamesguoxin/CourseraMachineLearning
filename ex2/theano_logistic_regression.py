import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
rng = np.random

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        # input: theano.tensor.TensorType
        #        symbolic variable that describes the input of the architucture
        #        (one mini-batch)
        self.W = theano.shared(value=np.zeros(
                                     (n_in, n_out), 
                                     dtype=theano.config.floatX),
                               name="W",
                               borrow=True)
        self.b = theano.shared(value=np.zeros(
                                     (n_out,),
                                     dtype=theano.config.floatX),
                               name="b",
                               borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        # y.shape[0] is the number of rows of y
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                    'y should have the same shape as self.y_pred',
                    ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def main():
    pass

if __name__ == "__main__":
    main()
