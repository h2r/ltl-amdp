# Python imports.
from __future__ import print_function
from collections import defaultdict
from copy import deepcopy

# Other imports.
from simple_rl.mdp.oomdp.OOMDPStateClass import OOMDPState
from simple_rl.mdp.oomdp.OOMDPClass import OOMDP
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject
from simple_rl.planning import ValueIteration
from simple_rl.amdp.AMDPTaskNodesClass import NonPrimitiveAbstractTask, RootTaskNode

class TaxiL1State(OOMDPState):
    def __init__(self, agent_obj, passenger_obj, is_terminal=False):
        '''
        Args:
            agent_obj (OOMDPObject)
            passenger_obj (OOMDPObject): Assuming single passenger domain for now
            is_terminal (bool)
        '''
        self.agent_obj = agent_obj
        self.passenger_obj = passenger_obj
        OOMDPState.__init__(self, {'agent': [agent_obj], 'passenger': [passenger_obj]}, is_terminal=is_terminal)

    def __eq__(self, other):
        return isinstance(other, TaxiL1State) and self.agent_obj.attributes == other.agent_obj.attributes and \
               self.passenger_obj.attributes == other.passenger_obj.attributes

    def __hash__(self):
        agent = self.agent_obj
        passenger = self.passenger_obj
        return hash(tuple((agent['current_color'], agent['has_passenger'],
                           passenger['current_color'], passenger['dest_color'])))

class TaxiL1GroundedAction(NonPrimitiveAbstractTask):
    def __init__(self, l1_action_string, subtasks, lower_domain):
        '''
        Args:
             l1_action_string (str): string representing action in l1 domain
             subtasks (list):  list of PrimitiveAbstractTask objects
             lower_domain (OOMDP): L0 MDP
        '''
        self.action = l1_action_string
        self.goal_parameter = TaxiL1GroundedAction._extract_goal_parameter_from_action(l1_action_string)
        self.is_navigation_task = self.goal_parameter.lower() in TaxiL1OOMDP.COLORS
        self.is_pickup_task = 'pickup' in self.goal_parameter.lower()
        self.is_dropoff_task = 'dropoff' in self.goal_parameter.lower()
        tf, rf = self._terminal_function, self._reward_function
        self.l0_domain = lower_domain
        NonPrimitiveAbstractTask.__init__(self, l1_action_string, subtasks, tf, rf)

    def _terminal_function(self, state):
        '''
        Args:
            state: (OOMDPState) l0_state
        Returns:
            terminal (bool): whether l0_state is terminal in l1_domain
        '''
        from simple_rl.amdp.abstr_domains.taxi.AbstractTaxiStateMapperClass import AbstractTaxiL1StateMapper
        def _color_terminal_function(s, goal_color):
            return s.agent_obj['current_color'] == goal_color
        def _pickup_terminal_function(s):
            return s.agent_obj['has_passenger'] and s.agent_obj['current_color'] == s.passenger_obj['current_color']
        def _dropoff_terminal_function(s):
            return (not s.agent_obj['has_passenger']) and s.agent_obj['current_color'] == \
                   s.passenger_obj['dest_color'] == s.passenger_obj['current_color']

        # The L1 domain should only have to reason about L1 states
        state_mapper = AbstractTaxiL1StateMapper(self.l0_domain)
        projected_state = state_mapper.map_state(state)

        if self.is_navigation_task:
            return _color_terminal_function(projected_state, self.goal_parameter)
        if self.is_pickup_task:
            return _pickup_terminal_function(projected_state)
        if self.is_dropoff_task:
            return _dropoff_terminal_function(projected_state)
        raise ValueError('goal_paramater {} did not fall into expected categories'.format(self.goal_parameter))

    def _reward_function(self, state):
        return 1. if self._terminal_function(state) else 0.

    @classmethod
    def _extract_goal_parameter_from_action(cls, l1_action_str):
        action = l1_action_str
        if 'to' in l1_action_str:
            action = l1_action_str.split('to')[1].lower()
        return action

class TaxiRootGroundedAction(RootTaskNode):
    def __init__(self, action_str, subtasks, l1_domain, terminal_func, reward_func):
        self.action = action_str
        dest_color = l1_domain.init_state.get_first_obj_of_class('passenger')['dest_color']
        self.goal_state = TaxiL1OOMDP.create_goal_state(dest_color)
        RootTaskNode.__init__(self, action_str, subtasks, l1_domain, terminal_func, reward_func)

class TaxiL1OOMDP(OOMDP):
    COLORS = ['red', 'green', 'blue', 'yellow']
    ACTIONS = ['l1_pickup', 'l1_dropoff', 'toRed', 'toGreen', 'toBlue', 'toYellow']
    CLASSES = ['agent', 'passenger']

    def __init__(self, agent_color='red', passenger_color='blue', passenger_dest_color='green'):
        agent = {'current_color': agent_color, 'has_passenger': 0}
        passengers = [{'current_color': passenger_color, 'dest_color': passenger_dest_color}]

        agent_obj = OOMDPObject(attributes=agent, name='agent')
        passenger_objs = self._make_oomdp_objs_from_list_of_dict(passengers, 'passenger')

        init_state = TaxiL1OOMDP._create_state(agent_obj, passenger_objs)

        self.goal_state = TaxiL1OOMDP.create_goal_state(passenger_dest_color)
        self.terminal_func = lambda state: state == self.goal_state

        OOMDP.__init__(self, TaxiL1OOMDP.ACTIONS, self._transition_func, self._reward_func, init_state)

    @classmethod
    def _create_state(cls, agent_obj, passenger_objs, is_terminal=False):
        return TaxiL1State(agent_obj, passenger_objs[0], is_terminal=is_terminal)

    @classmethod
    def create_goal_state(cls, dest_color):
        goal_agent = {'current_color': dest_color, 'has_passenger': 0}
        goal_passengers = [{'current_color': dest_color, 'dest_color': dest_color}]
        goal_agent_obj = OOMDPObject(attributes=goal_agent, name='agent')
        goal_passenger_objs = OOMDP._make_oomdp_objs_from_list_of_dict(goal_passengers, 'passenger')
        return TaxiL1OOMDP._create_state(goal_agent_obj, goal_passenger_objs, is_terminal=True)

    def _reward_func(self, state, action):
        if self._is_goal_state_action(state, action):
            return 1.
        return 0.

    def _is_goal_state_action(self, state, action):
        if state == self.goal_state:
            return False
        return self._transition_func(state, action) == self.goal_state

    def _transition_func(self, state, action):
        if state.is_terminal():
            return state

        next_state = deepcopy(state)
        if action == 'toRed':
            next_state.agent_obj['current_color'] = 'red'
            if state.agent_obj['has_passenger']:
                next_state.passenger_obj['current_color'] = 'red'
        if action == 'toGreen':
            next_state.agent_obj['current_color'] = 'green'
            if state.agent_obj['has_passenger']:
                next_state.passenger_obj['current_color'] = 'green'
        if action == 'toBlue':
            next_state.agent_obj['current_color'] = 'blue'
            if state.agent_obj['has_passenger']:
                next_state.passenger_obj['current_color'] = 'blue'
        if action == 'toYellow':
            next_state.agent_obj['current_color'] = 'yellow'
            if state.agent_obj['has_passenger']:
                next_state.passenger_obj['current_color'] = 'yellow'
        if action == 'l1_pickup':
            if (not state.agent_obj['has_passenger']) and (state.agent_obj['current_color'] ==
                state.passenger_obj['current_color']):
                next_state.agent_obj['has_passenger'] = True
        if action == 'l1_dropoff':
            if state.agent_obj['current_color'] == state.passenger_obj['dest_color'] and state.agent_obj['has_passenger']:
                next_state.agent_obj['has_passenger'] = False

        if next_state == self.goal_state:
            next_state.set_terminal(True)

        return next_state

# -----------------------------------
# Debug functions
# -----------------------------------

def debug_l1_taxi():
    def get_l1_policy(oomdp=None):
        if oomdp is None:
            oomdp = TaxiL1OOMDP()
        vi = ValueIteration(oomdp, sample_rate=1)
        vi.run_vi()

        policy = defaultdict()
        action_seq, state_seq = vi.plan(oomdp.init_state)

        print('Plan for {}:'.format(oomdp))
        for i in range(len(action_seq)):
            print("\tpi[{}] -> {}\n".format(state_seq[i], action_seq[i]))
            policy[state_seq[i]] = action_seq[i]

        return policy

    mdp = TaxiL1OOMDP()
    pi = get_l1_policy(mdp)
    return pi