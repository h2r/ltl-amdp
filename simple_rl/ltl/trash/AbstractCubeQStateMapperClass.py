# Other imports.
from simple_rl.amdp.AMDPStateMapperClass import AMDPStateMapper
from simple_rl.amdp.abstr_domains.grid_world.AbstractGridWorldMDPClass import FourRoomL1State
from simple_rl.ltl.AMDP.AbstractCubeQMDPClass import CubeL1State, CubeL2State

class AbstractCubeL1StateMapper(AMDPStateMapper): # TODO: Modify
    def __init__(self, l0_domain):
        AMDPStateMapper.__init__(self, l0_domain)

    def map_state(self, l0_state):
        '''
        Args:
            l0_state (OOMDPState): L0 RoomCubeMDP
        Returns:
            projected_state (TaxiL1State): Mapping of state into L1 space
        '''
        l0_location = (l0_state.x, l0_state.y, l0_state.z)
        room = self.lower_domain.get_room_numbers(l0_location)[0]
        return CubeL1State(room, l0_state.q)

class AbstractCubeL2StateMapper(AMDPStateMapper): # TODO: Modify
    def __init__(self, l1_domain):
        AMDPStateMapper.__init__(self, l1_domain)

    def map_state(self, l1_state):
        '''
        Args:
            l0_state (OOMDPState): L1 CubeL1MDP
        Returns:
            projected_state (TaxiL1State): Mapping of state into L2 space
        '''

        floor = self.lower_domain.get_floor_numbers(l1_state.agent_in_room_number)[0]
        return CubeL2State(floor, l1_state.q)


