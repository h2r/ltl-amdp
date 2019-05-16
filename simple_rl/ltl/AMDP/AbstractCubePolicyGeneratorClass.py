# Other imports.
#from simple_rl.tasks.four_room.FourRoomMDPClass import FourRoomMDP
#from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
#from simple_rl.amdp.abstr_domains.grid_world.AbstractGridWorldMDPClass import FourRoomL1MDP
from simple_rl.amdp.AMDPPolicyGeneratorClass import AMDPPolicyGenerator
#from simple_rl.amdp.abstr_domains.grid_world.AbstractGridWorldStateMapperClass import AbstractGridWorldL1StateMapper

from simple_rl.ltl.AMDP.RoomCubeMDPClass import RoomCubeMDP
from simple_rl.ltl.AMDP.CubeStateClass import CubeState
from simple_rl.ltl.AMDP.AbstractCubeMDPClass import CubeL1MDP, CubeL2MDP
from simple_rl.ltl.AMDP.AbstractCubeStateMapperClass import AbstractCubeL1StateMapper, AbstractCubeL2StateMapper

class CubeL2PolicyGenerator(AMDPPolicyGenerator):
    def __init__(self, l1MDP, state_mapper, verbose=False, env_file =[], constraints = {}, ap_maps = {}):
        '''
        Args:
            l1MDP (FourRoomMDP): lower domain
            state_mapper (AbstractGridWorldL1StateMapper): to map l0 states to l1 domain
            verbose (bool): debug mode
        '''
        self.domain = l1MDP
        self.verbose = verbose
        self.state_mapper = state_mapper
        self.env_file = env_file

    def generate_policy(self, l2_state, grounded_action):
        '''
        Args:
            l1_state (FourRoomL1State): generate policy in l1 domain starting from l1_state
            grounded_action (FourRoomRootGroundedAction): TaskNode above defining the subgoal for current MDP
        '''
        mdp = CubeL2MDP(l2_state.agent_on_floor_number, env_file=self.env_file,
                        constraints=grounded_action.goal_constraints,
                        ap_maps=grounded_action.ap_maps)
        return self.get_policy(mdp, verbose=self.verbose, max_iterations=50, horizon=100)

    def generate_abstract_state(self, l1_state):
        return self.state_mapper.map_state(l1_state)

class CubeL1PolicyGenerator(AMDPPolicyGenerator):
    def __init__(self, l0MDP, state_mapper, verbose=False, env_file = [], constraints = {}, ap_maps = {}):
        '''
        Args:
            l0MDP (FourRoomMDP): lower domain
            state_mapper (AbstractGridWorldL1StateMapper): to map l0 states to l1 domain
            verbose (bool): debug mode
        '''
        self.domain = l0MDP
        self.verbose = verbose
        self.state_mapper = state_mapper
        self.env_file = env_file
        self.constraints = constraints
        self.ap_maps = ap_maps

    def generate_policy(self, l1_state, grounded_action):
        '''
        Args:
            l1_state (FourRoomL1State): generate policy in l1 domain starting from l1_state
            grounded_action (FourRoomRootGroundedAction): TaskNode above defining the subgoal for current MDP
        '''
        #destination_locations = self.grounded_action.l1_domain.
        #.floor_to_rooms[grounded_action.goal_state.agent_on_floor_number]
        mdp = CubeL1MDP(l1_state.agent_in_room_number, env_file=self.env_file,
                        constraints=grounded_action.goal_constraints,
                        ap_maps=grounded_action.ap_maps)
        return self.get_policy(mdp, verbose=self.verbose, max_iterations=50, horizon=100)

    def generate_abstract_state(self, l0_state):
        return self.state_mapper.map_state(l0_state)

class CubeL0PolicyGenerator(AMDPPolicyGenerator):
    def __init__(self, l0_domain, verbose=False, env_file = []):
        self.domain = l0_domain
        self.verbose = verbose
        self.env_file = env_file

    def generate_policy(self, state, grounded_task):
        '''
        Args:
             state (GridWorldState): plan in L0 starting from state
             grounded_task (FourRoomL1GroundedAction): L1 TaskNode defining L0 subgoal
        '''
#        destination_locations = self.domain.room_to_locs[grounded_task.goal_state.agent_in_room_number]
        init_location = (state.x, state.y, state.z)
        mdp = RoomCubeMDP(init_loc=init_location, env_file=self.env_file,
                          constraints=grounded_task.goal_constraints, ap_maps=grounded_task.ap_maps)
        return self.get_policy(mdp, verbose=self.verbose, max_iterations=50, horizon=100) # 500, 100
