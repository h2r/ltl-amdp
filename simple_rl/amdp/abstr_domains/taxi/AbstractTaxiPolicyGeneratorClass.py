# Python imports.
from collections import defaultdict

# Other imports.
from simple_rl.tasks.taxi.TaxiOOMDPClass import TaxiOOMDP
from simple_rl.amdp.AMDPPolicyGeneratorClass import AMDPPolicyGenerator
from simple_rl.amdp.abstr_domains.taxi.AbstractTaxiMDPClass import TaxiL1OOMDP, TaxiL1State, TaxiL1GroundedAction, TaxiRootGroundedAction

class TaxiL1PolicyGenerator(AMDPPolicyGenerator):
    def __init__(self, l0MDP, state_mapper, verbose=False):
        self.domain = l0MDP
        self.verbose = verbose
        self.state_mapper = state_mapper

    def generate_policy(self, l1_state, grounded_action):
        '''
        Args:
            l1_state (TaxiL1State)
            grounded_action (TaxiRootGroundedAction)

        Returns:
            policy (defaultdict)
        '''
        agent_color = l1_state.agent_obj['current_color']
        passenger_color = l1_state.passenger_obj['current_color']
        passenger_dest_color = grounded_action.goal_state.passenger_obj['dest_color']
        mdp = TaxiL1OOMDP(agent_color=agent_color, passenger_color=passenger_color,
                          passenger_dest_color=passenger_dest_color)
        return self.get_policy(mdp)

    def generate_abstract_state(self, l0_state):
        return self.state_mapper.map_state(l0_state)

class TaxiL0PolicyGenerator(AMDPPolicyGenerator):
    def __init__(self, l0_domain, verbose=False):
        self.domain = l0_domain
        self.verbose = verbose

    def generate_policy(self, l0_state, l1_grounded_task):
        policy = defaultdict()
        if l1_grounded_task.is_navigation_task:
            agent = l0_state.get_first_obj_of_class('agent')
            passenger = l0_state.get_first_obj_of_class('passenger')
            destination_color = l1_grounded_task.goal_parameter
            destination_location = self.domain.location_for_color(destination_color)
            agent_dict = {'x': agent['x'], 'y': agent['y'], 'has_passenger': agent['has_passenger']}

            # The exact attribute values of passenger are irrelavant
            # We solve the navigation MDP by leveraging goal_loc = destination_location
            passenger_dict = {'x': passenger['x'], 'y': passenger['y'], 'in_taxi': passenger['in_taxi'],
                              'dest_x': passenger['dest_x'], 'dest_y': passenger['dest_y']}

            mdp = TaxiOOMDP(self.domain.width, self.domain.height, agent_dict, [], [passenger_dict],
                            goal_loc=destination_location)

            return self.get_policy(mdp)

        elif l1_grounded_task.is_pickup_task:
            policy[l0_state] = 'pickup'

        elif l1_grounded_task.is_dropoff_task:
            policy[l0_state] = 'dropoff'

        return policy
