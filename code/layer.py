"""All the layer functions go here.
"""

from __future__ import division, print_function, absolute_import
import numpy as np


class FullyConnected(object):
    """Fully connected layer 'y = Wx + b'.

    Arguments:
        shape(tuple): the shape of the fully connected layer. shape[0] is the
            output size and shape[1] is the input size.

    Attributes:
        W(np.array): the weights of the fully connected layer. An n-by-m matrix
            where m is the input size and n is the output size.
        b(np.array): the biases of the fully connected layer. A n-by-1 vector
            where n is the output size.

    """

    def __init__(self, shape):
        
        self.W = np.random.randn(shape[0],shape[1])
        self.b = np.random.randn(shape[0], 1)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x(np.array): the input of the layer.

        Returns:
            The output of the layer.

        """

        # TODO: Forward code
        linear_combination = np.dot(self.W,x)+self.b
        return(linear_combination)

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
        respect to the input.

        Args:
            x(np.array): the input of the layer.
            dv_y(np.array): The derivative of the loss with respect to the
                output.

        Returns:
            dv_x(np.array): The derivative of the loss with respect to the
                input.
            dv_W(np.array): The derivative of the loss with respect to the
                weights.
            dv_b(np.array): The derivative of the loss with respect to the
                biases.

        """

        # TODO: Backward code
        dv_b = dv_y
        dv_W = np.dot(x,dv_y.T).T
        dv_x = np.dot(self.W.T,dv_y)
        out = [dv_x,dv_W,dv_b]
        return(out)
    
    def update(self,W_new,b_new):
        self.W = W_new
        self.b = b_new


class Sigmoid(object):
    """Sigmoid function 'y = 1 / (1 + exp(-x))'

    """
    def __init__(self):    
        self.W=None
        self.b=None
    
    def forward(self, x):
        """Compute the layer output.

        Args:
            x(np.array): the input of the layer.

        Returns:
            The output of the layer.

        """

        # TODO: Forward code
        z = 1.0/(1.0+np.exp(-x))
        return(z)

    def backward(self, x, dv_y):
        """Compute the gradient with respect to the input.

        Args:
            x(np.array): the input of the layer.
            dv_y(np.array): The derivative of the loss with respect to the
                output.

        Returns:
            The derivative of the loss with respect to the input.

        """
        
        y = self.forward(x)
        dv_x = y*(1-y)*dv_y
        dv_w = False
        dv_b = False
        out = [dv_x,dv_w,dv_b]
        return(out)
    
    
class tanh(object):
    """Rectilinear unit : 
       
       f(x) = {(1-e^(−2x))}/{1+e^(−2x)}

    """
    def __init__(self):    
        self.W=None
        self.b=None

    def forward(self, x):
        """Compute the layer output.

        Args:
            x(np.array): the input of the layer.

        Returns:
            The output of the layer.

        """
        z = (1-np.exp(-2*x))/(1+np.exp(-2*x))
        return(z)

    def backward(self, x, dv_y):
        """Compute the gradient with respect to the input.

        Args:
            x(np.array): the input of the layer.
            dv_y(np.array): The derivative of the loss with respect to the
                output.

        Returns:
            The derivative of the loss with respect to the input.

        """
        
        z = self.forward(x)
        dv_x = (1-np.square(z))*dv_y
        dv_w = False
        dv_b = False
        out = [dv_x,dv_w,dv_b]
        return(out)