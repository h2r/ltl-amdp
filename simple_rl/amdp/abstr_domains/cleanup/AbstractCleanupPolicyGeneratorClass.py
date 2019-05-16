# Python imports.
from collections import defaultdict
import copy

# Other imports.
from simple_rl.amdp.AMDPPolicyGeneratorClass import AMDPPolicyGenerator
from simple_rl.amdp.abstr_domains.cleanup.AbstractCleanupL1StateClass import CleanupL1State
from simple_rl.amdp.abstr_domains.cleanup.AbstractCleanupMDPClass import CleanupL1MDP
from simple_rl.tasks.cleanup.cleanup_state import CleanUpState
from simple_rl.tasks.cleanup.CleanupMDPClass import CleanUpMDP

class CleanupL1PolicyGenerator(AMDPPolicyGenerator):
    def __init__(self, l0_domain, state_mapper, verbose=False):
        self.domain = l0_domain
        self.state_mapper = state_mapper
        self.verbose = verbose

    def generate_policy(self, l1_state, grounded_task):
        '''
        Args:
            l1_state (CleanupL1State)
            grounded_task (CleanupRootGroundedAction)

        Returns:
            policy (defaultdict)
        '''
        mdp = CleanupL1MDP(self.domain)
        return self.get_policy(mdp, verbose=True)

    def generate_abstract_state(self, l0_state):
        return self.state_mapper.map_state(l0_state)

class CleanupL0PolicyGenerator(AMDPPolicyGenerator):
    def __init__(self, l0_domain, verbose=False):
        '''
        Args:
            l0_domain (CleanUpMDP)
            verbose (bool)
        '''
        self.domain = l0_domain
        self.verbose = verbose

    def generate_policy(self, l0_state, grounded_task):
        '''
        Args:
            l0_state (CleanUpState)
            grounded_task (CleanupL1GroundedAction)

        Returns:
             policy (defaultdict): optimal policy in mdp defined by `grounded_task`
        '''
        mdp = copy.deepcopy(self.domain)
        mdp.terminal_func = grounded_task.terminal_func
        mdp.reward_func = grounded_task.reward_func
        mdp.init_state = l0_state
        return self.get_policy(mdp)
