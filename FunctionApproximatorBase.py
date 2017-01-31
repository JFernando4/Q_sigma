import abc


class FunctionApproximatorBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """ Initializes the weights and parameters used for the function approximator """

    @abc.abstractmethod
    def update(self, state, action, value):
        """ Updates the function approximator """

    @abc.abstractmethod
    def get_value(self, value, action):
        """ Returns the approximation to the action-value or the state-value function """
        return

    @abc.abstractmethod
    def get_alpha_factor(self):
        """ Returns the adjusting factor for alpha. It depends on each different function approximator and can be
            set to 1 """
        return