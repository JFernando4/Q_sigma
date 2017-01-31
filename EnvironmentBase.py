import abc
# Note: I avoid using the word Mdp because an environment can be either deterministic
# or random. This implementation doesn't differentiate between them.


class EnvironmentBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def init_state(self):
        """Returns the initial state of the MDP"""
        return

    @abc.abstractmethod
    def do_action(self, S, A):
        """"Given an action, it returns the new state and the corresponding reward"""
        return

    @abc.abstractmethod
    def get_num_actions(self):
        """ Returns the number of actions available to the agent """
        return
