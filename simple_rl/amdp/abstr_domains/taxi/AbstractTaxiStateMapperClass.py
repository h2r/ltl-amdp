# Other imports.
from simple_rl.amdp.AMDPStateMapperClass import AMDPStateMapper
from simple_rl.mdp.oomdp.OOMDPStateClass import OOMDPState
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject
from simple_rl.amdp.abstr_domains.taxi.AbstractTaxiMDPClass import TaxiL1State

class AbstractTaxiL1StateMapper(AMDPStateMapper):
    def __init__(self, l0_domain):
        AMDPStateMapper.__init__(self, l0_domain)
        self.l0_domain = l0_domain

    def map_state(self, l0_state):
        '''
        Args:
            l0_state (OOMDPState): L0 Taxi State
        Returns:
            projected_state (TaxiL1State): Mapping of state into L1 space
        '''
        agent = l0_state.get_first_obj_of_class('agent')
        passenger = l0_state.get_first_obj_of_class('passenger')
        agent_location = agent['x'], agent['y']
        passenger_location = passenger['x'], passenger['y']
        destination = passenger['dest_x'], passenger['dest_y']
        agent_color = self.l0_domain.color_for_location(agent_location)
        passenger_color = self.l0_domain.color_for_location(passenger_location)
        passenger_dest_color = self.l0_domain.color_for_location(destination)

        agent_dict = {'current_color': agent_color, 'has_passenger': agent['has_passenger']}
        passengers = [{'current_color': passenger_color, 'dest_color': passenger_dest_color}]
        agent_obj = OOMDPObject(attributes=agent_dict, name='agent')
        passenger_objs = self.l0_domain._make_oomdp_objs_from_list_of_dict(passengers, 'passenger')

        return TaxiL1State(agent_obj, passenger_objs[0])