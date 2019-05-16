# Other imports.
from simple_rl.tasks.four_room.FourRoomMDPClass import FourRoomMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from simple_rl.amdp.abstr_domains.grid_world.AbstractGridWorldMDPClass import FourRoomL1MDP
from simple_rl.amdp.AMDPPolicyGeneratorClass import AMDPPolicyGenerator
from simple_rl.amdp.abstr_domains.grid_world.AbstractGridWorldStateMapperClass import AbstractGridWorldL1StateMapper

class GridWorldL1PolicyGenerator(AMDPPolicyGenerator):
    def __init__(self, l0MDP, state_mapper, verbose=False):
        '''
        Args:
            l0MDP (FourRoomMDP): lower domain
            state_mapper (AbstractGridWorldL1StateMapper): to map l0 states to l1 domain
            verbose (bool): debug mode
        '''
        self.domain = l0MDP
        self.verbose = verbose
        self.state_mapper = state_mapper

    def generate_policy(self, l1_state, grounded_action):
        '''
        Args:
            l1_state (FourRoomL1State): generate policy in l1 domain starting from l1_state
            grounded_action (FourRoomRootGroundedAction): TaskNode above defining the subgoal for current MDP
        '''
        mdp = FourRoomL1MDP(l1_state.agent_in_room_number, grounded_action.goal_state.agent_in_room_number)
        return self.get_policy(mdp, verbose=True)

    def generate_abstract_state(self, l0_state):
        return self.state_mapper.map_state(l0_state)

class GridWorldL0PolicyGenerator(AMDPPolicyGenerator):
    def __init__(self, l0_domain, verbose=False):
        self.domain = l0_domain
        self.verbose = verbose

    def generate_policy(self, state, grounded_task):
        '''
        Args:
             state (GridWorldState): plan in L0 starting from state
             grounded_task (FourRoomL1GroundedAction): L1 TaskNode defining L0 subgoal
        '''
        destination_locations = self.domain.room_to_locs[grounded_task.goal_state.agent_in_room_number]
        init_location = (state.x, state.y)
        mdp = FourRoomMDP(self.domain.width, self.domain.height, init_loc=init_location, goal_locs=destination_locations)
        return self.get_policy(mdp, verbose=True)
