# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.MDPClass import MDP

class AMDPStateMapper(object):
    '''
    Abstract AMDP state mapper class.
    This class will project a state from a lower domain to
    the domain represented by the current level of hierarchy
    '''
    def __init__(self, lower_domain):
        '''
        Args:
            lower_domain (MDP): MDP lower in the hierarchy from which
            we are going to project
        '''
        self.lower_domain = lower_domain

    def map_state(self, state):
        '''
        Args:
            state (State): state in the lower domain
        Returns:
            projected_state (State): state in the current (higher) level
        '''
        pass