# Python imports.
from __future__ import print_function
from collections import defaultdict
import re

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.MDPClass import MDP
from simple_rl.planning import ValueIteration
from simple_rl.amdp.AMDPTaskNodesClass import NonPrimitiveAbstractTask, RootTaskNode

class FourRoomL1State(State):
    def __init__(self, room_number, is_terminal=False):
        State.__init__(self, data=[room_number], is_terminal=is_terminal)
        self.agent_in_room_number = room_number

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return 'Agent in room {}'.format(self.agent_in_room_number)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, FourRoomL1State) and self.agent_in_room_number == other.agent_in_room_number

class FourRoomL1GroundedAction(NonPrimitiveAbstractTask):
    def __init__(self, l1_action_string, subtasks, lowerDomain):
        self.action = l1_action_string
        goal_room = self.extract_goal_room(self.action)
        self.goal_state = FourRoomL1State(goal_room, is_terminal=True)
        tf, rf = self._terminal_function, self._reward_function
        self.l0_domain = lowerDomain
        NonPrimitiveAbstractTask.__init__(self, l1_action_string, subtasks, tf, rf)

    @classmethod
    def extract_goal_room(cls, action):
        room_numbers = re.findall(r'\d+', action)
        if len(room_numbers) == 0:
            raise ValueError('unable to extract room number from L1Action {}'.format(action))
        return int(room_numbers[0])

    def _terminal_function(self, state):
        if type(state) == FourRoomL1State:
            return state == self.goal_state
        room_number = self._room_number(state)
        assert type(room_number) == int, 'Room number {} not convertible to int'.format(room_number)
        return FourRoomL1State(room_number) == self.goal_state

    def _reward_function(self, state):
        if type(state) == FourRoomL1State:
            return 1. if state == self.goal_state else 0.
        room_number = self._room_number(state)
        assert type(room_number) == int, 'Room number {} not convertible to int'.format(room_number)
        return 1. if FourRoomL1State(room_number) == self.goal_state else 0.

    def _room_number(self, state):
        return self.l0_domain.get_room_numbers((int(state.x), int(state.y)))[0]

class FourRoomRootGroundedAction(RootTaskNode):
    def __init__(self, action_str, subtasks, l1_domain, terminal_func, reward_func):
        self.action = 'Root_' + action_str
        self.goal_state = FourRoomL1State(FourRoomL1GroundedAction.extract_goal_room(action_str), is_terminal=True)

        RootTaskNode.__init__(self, self.action, subtasks, l1_domain, terminal_func, reward_func)

class FourRoomL1MDP(MDP):
    ACTIONS = ['toRoom1', 'toRoom2', 'toRoom3', 'toRoom4']
    def __init__(self, starting_room=1, goal_room=4, gamma=0.99):
        initial_state = FourRoomL1State(starting_room)
        self.goal_state = FourRoomL1State(goal_room, is_terminal=True)
        self.terminal_func = lambda state: state == self.goal_state

        MDP.__init__(self, FourRoomL1MDP.ACTIONS, self._transition_func, self._reward_func, init_state=initial_state,
                     gamma=gamma)

    def _reward_func(self, state, action):
        if self._is_goal_state_action(state, action):
            return 1.0
        return 0.0

    def _is_goal_state_action(self, state, action):
        if state == self.goal_state:
            return False

        return self._transition_func(state, action) == self.goal_state

    def _transition_func(self, state, action):
        if state.is_terminal():
            return state

        current_room = state.agent_in_room_number
        next_state = None
        if current_room == 1:
            if action == 'toRoom2':
                next_state = FourRoomL1State(2)
            if action == 'toRoom3':
                next_state = FourRoomL1State(3)
        if current_room == 2:
            if action == 'toRoom1':
                next_state = FourRoomL1State(1)
            if action == 'toRoom4':
                next_state = FourRoomL1State(4)
        if current_room == 3:
            if action == 'toRoom4':
                next_state = FourRoomL1State(4)
            if action == 'toRoom1':
                next_state = FourRoomL1State(1)
        if current_room == 4:
            if action == 'toRoom2':
                next_state = FourRoomL1State(2)
            if action == 'toRoom3':
                next_state = FourRoomL1State(3)

        if next_state is None:
            next_state = state

        if next_state == self.goal_state:
            next_state.set_terminal(True)

        return next_state

    def __str__(self):
        return 'AbstractFourRoomMDP: InitState: {}, GoalState: {}'.format(self.init_state, self.goal_state)

    @classmethod
    def action_for_room_number(cls, room_number):
        for action in cls.ACTIONS:
            if str(room_number) in action:
                return action
        raise ValueError('unable to find action corresponding to room {}'.format(room_number))

# -----------------------------------
# Debug functions
# -----------------------------------

def debug_l1_grid_world():
    def get_l1_policy(start_room=None, goal_room=None, mdp=None):
        if mdp is None:
            mdp = FourRoomL1MDP(start_room, goal_room)
        vi = ValueIteration(mdp)
        vi.run_vi()

        policy = defaultdict()
        action_seq, state_seq = vi.plan(mdp.init_state)

        print('Plan for {}:'.format(mdp))
        for i in range(len(action_seq)):
            print("\tpi[{}] -> {}".format(state_seq[i], action_seq[i]))
            policy[state_seq[i]] = action_seq[i]
        return policy
    policy = get_l1_policy(1, 4)
