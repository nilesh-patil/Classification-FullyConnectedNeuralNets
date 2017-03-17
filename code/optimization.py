"""All the optimization methods go here.

"""

from __future__ import division, print_function, absolute_import
import random
import numpy as np


class SGD(object):
    """Mini-batch stochastic gradient descent.

    Attributes:
        learning_rate(float): the learning rate to use.
        batch_size(int): the number of samples in a mini-batch.

    """

    def __init__(self, learning_rate, batch_size):
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size

    def __has_parameters(self, layer):
        return hasattr(layer, "W")

    def compute_gradient(self, x, y, graph, loss):
        """ Compute the gradients of network parameters (weights and biases)
        using backpropagation.

        Args:
            x(np.array): the input to the network.
            y(np.array): the ground truth of the input.
            graph(obj): the network structure.
            loss(obj): the loss function for the network.

        Returns:
            dv_Ws(list): a list of gradients of the weights.
            dv_bs(list): a list of gradients of the biases.

        """

        # TODO: Backpropagation code
        dv_Ws = []
        dv_bs = []

        x_l = [x.copy()]
        y_pred = Network(graph).inference(x)
        error = loss.forward(y_pred,y)
        dv_o = loss.backward(y_pred,y)
        n_layer = len(graph.layers)
        stage = np.arange(n_layer)

        for layer_number in stage:
            layer = graph.layers[layer_number]
            x_l = x_l + [layer.forward(x_l[layer_number])]

        for layer_number in stage[::-1]:
            layer = graph.layers[layer_number]
            dv  = layer.backward(x=x_l[layer_number],dv_y=dv_o)
            dv_o = dv[0]
            dv_Ws = [dv[1]]+dv_Ws
            dv_bs = [dv[2]]+dv_bs

        out = [dv_Ws,dv_bs]
        return(out)

    def optimize(self, graph, loss, training_data):
        """ Perform SGD on the network defined by 'graph' using
        'training_data'.

        Args:
            graph(obj): a 'Graph' object that defines the structure of a
                neural network.
            loss(obj): the loss function for the network.
            training_data(list): a list of tuples ``(x, y)`` representing the
                training inputs and the desired outputs.

        """

        # Network parameters
        # Ws = [layer.W for layer in graph if self.__has_parameters(layer)]
        # bs = [layer.b for layer in graph if self.__has_parameters(layer)]

        # Shuffle the data to make sure samples in each batch are not
        # correlated
        random.shuffle(training_data)
        n = len(training_data)

        batches = [
            training_data[k:k + self.batch_size]
            for k in xrange(0, n, self.batch_size)
        ]

        # TODO: SGD code
        for batch in batches:

            dv_Ws=[]
            dv_bs=[]

            for x,y in batch:
                dv_Wi,dv_bi = self.compute_gradient(x=x,y=y,graph=graph,loss=loss)
                dv_Ws += [dv_Wi]
                dv_bs += [dv_bi]

            dv_Wsum = np.sum(dv_Ws,axis=0)
            dv_bsum = np.sum(dv_bs,axis=0)
            n_layer = len(graph.layers)

            for layer_number in np.arange(n_layer):
                if graph.layers[layer_number].W is not None:
                    graph.layers[layer_number].W -= (self.learning_rate*dv_Wsum[layer_number])/self.batch_size
                    graph.layers[layer_number].b -= (self.learning_rate*dv_bsum[layer_number])/self.batch_size
