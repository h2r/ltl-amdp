# Other imports.
from simple_rl.amdp.AMDPStateMapperClass import AMDPStateMapper
from simple_rl.amdp.abstr_domains.grid_world.AbstractGridWorldMDPClass import FourRoomL1State

class AbstractGridWorldL1StateMapper(AMDPStateMapper):
    def __init__(self, l0_domain):
        AMDPStateMapper.__init__(self, l0_domain)

    def map_state(self, l0_state):
        '''
        Args:
            l0_state (OOMDPState): L0 Taxi State
        Returns:
            projected_state (TaxiL1State): Mapping of state into L1 space
        '''
        l0_location = (l0_state.x, l0_state.y)
        room = self.lower_domain.get_room_numbers(l0_location)[0]
        return FourRoomL1State(room)
