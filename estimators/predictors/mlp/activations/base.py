from abc import ABC, abstractmethod


class BaseActivation(ABC):
    @staticmethod
    @abstractmethod
    def sigma(x):
        """Activation function"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def sigma_prime(x):
        """Derivative of the activation function"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_weight_init_std(fan_in: int, fan_out: int):
        """
        :param fan_in: number of input neurons of the layer.
        :param fan_out: number of output neurons of the layer.
        :return: the standard deviation that the weights of the layer should have.
        """
        raise NotImplementedError()
