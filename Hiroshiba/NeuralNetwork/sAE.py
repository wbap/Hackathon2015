# coding: utf-8

import random

import numpy

from SuperClass import SuperClass

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class SparseAutoEncoder(SuperClass):
    def __init__(self, n_in, n_hidden, lamda = 0.0001, rho = 0.01, beta = 3.0, W=None, bhid=None, bvis=None, n_epoch=20, batchsize=100, use_cuda=False):
        super().__init__(n_epoch, batchsize, use_cuda)
        
        # allocate symbolic variables for the data
        self.index = T.lscalar()    # index to a [mini]batch
        self.x = T.matrix('x')  # the data is presented as rasterized images

        rng = numpy.random.RandomState(None)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
        
        self.ae = TheanoSAE(rng, theano_rng, self.x, n_in, n_hidden, lamda, rho, beta)
        self.cost, self.updates = self.ae.get_cost_updates(0.0, 0.1)
        
        
    def train(self, x_data):
        shared_x = theano.shared(numpy.asarray(x_data, dtype=theano.config.floatX), borrow=True)
        
        train_ae = theano.function(
            [self.index],
            self.cost,
            updates=self.updates,
            givens={
                self.x: shared_x[self.index * self.batchsize: (self.index + 1) * self.batchsize]
            }
        )

        # go through training epochs
        for epoch in range(self.n_epoch):
            sum_cost = 0

            # go through training set
            n_train_batches = x_data.shape[0] // self.batchsize
            for batch_index in range(n_train_batches):
                cost = train_ae(batch_index)
                sum_cost += cost
            print('epoch:'+str(epoch)+' loss:' + str(sum_cost/x_data.shape[0]))


class TheanoSAE(object):
    """
    Sparse Auto-Encoder class using Theano. Based on the official Theano tutorial series.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        lamda = 0.0001,
        rho = 0.01,
        beta = 3.0,
        W=None,
        bhid=None,
        bvis=None
    ):
        """
        Initialize the class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ), the corruption level ( de-noising ),
        the lamda value ( weight decay parameter ), the rho value ( sparsity paramter ),
        and the beta value ( weight of sparsity term ). The constructor also
        receives symbolic variables for the input, weights and bias. Such a
        symbolic variables are useful when, for example the input is the result
        of some computations, or when weights are shared between the dA and an
        MLP layer. When dealing with SdAs this always happens, the dA on layer
        2 gets as input the output of the dA on layer 1, and the weights of the
        dA are used in the second stage of training to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type lamda: float32
        :param lamda:   the weight decay paramter that decreases the magnitude
                        of weights to help prevent overfitting

        :type rho: float32
        :param rho:  the sparsity parameter representing the ideal average
                    activation of each hidden neuron

        :type beta: float32
        :param beta:  the beta paramter that controls the weight of te sparsity
                        term in the overall cost function

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lamda = lamda
        self.rho = rho
        self.beta = beta

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """ This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """ Computes the reconstructed input given the values of the
        hidden layer
        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # Calculate cross-entropy cost (as alternative to MSE) of the reconstruction of the minibatch.

        weight_decay = 0.5 * self.lamda * (T.sum(T.mul(self.W, self.W)) + T.sum(T.mul(self.W_prime, self.W_prime)))
        # Calculate weight decay term to prevent overfitting

        rho_hat = T.sum(y, axis=1) / tilde_x.shape[1]
        KL_divergence = self.beta * T.sum(self.rho * T.log(self.rho / rho_hat) + (1-self.rho) * T.log((1 - self.rho)/(1-rho_hat)))
        # KL divergence sparsity term

        # Calculate overall errors
        cost = T.mean(L) + weight_decay + KL_divergence

        # Compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)

        # Generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)
