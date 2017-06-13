"""
Abstract class for layers
"""
from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forwards_pass(self, input_vector):
        """Forward propogation of input vector"""
        raise NotImplementedError

    @abstractmethod
    def backwards_pass(self, output_gradient):
        """Backpropogation to calculate input vector gradient"""
        raise NotImplementedError

    @abstractmethod
    def update_parameters(self, learning_rate):
        """updates weights and biases"""
        raise NotImplementedError





