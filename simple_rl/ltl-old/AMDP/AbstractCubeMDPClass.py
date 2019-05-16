# Python imports.
from __future__ import print_function
from collections import defaultdict
import re as re1

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.MDPClass import MDP
from simple_rl.planning import ValueIteration
from simple_rl.amdp.AMDPTaskNodesClass import NonPrimitiveAbstractTask, RootTaskNode
from simple_rl.ltl.settings.build_cube_env_1 import build_cube_env

from sympy import *
import random

class CubeL2State(State):
    def __init__(self, floor_number, q, is_terminal=False):
        State.__init__(self, data=[floor_number,q], is_terminal=is_terminal)
        self.agent_on_floor_number = floor_number
        self.q = q # logic state

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return 'Agent on the floor {}, Q: {}'.format(self.agent_on_floor_number, self.q)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, CubeL2State) and self.agent_on_floor_number == other.agent_on_floor_number

class CubeL2GroundedAction(NonPrimitiveAbstractTask):
    def __init__(self, l2_action_string, subtasks, lowerDomain):
        self.action = l2_action_string
        self.goal_floor = self.extract_goal_floor(self.action)

        self.goal_constraints = {'goal': 'a', 'stay': '~a'}
        self.ap_maps = {'a': (2, 'state', self.goal_floor)}
        tf, rf = self._terminal_function, self._reward_function
        self.l1_domain = lowerDomain
        NonPrimitiveAbstractTask.__init__(self, l2_action_string, subtasks, tf, rf)

    @classmethod
    def extract_goal_floor(cls, action): # parser
        floor_numbers = re1.findall(r'\d+', action)
        if len(floor_numbers) == 0:
            raise ValueError('unable to extract floor number from L2Action {}'.format(action))
        return int(floor_numbers[0])

    def _terminal_function(self, state):

        return self.l1_domain.get_floor_numbers(state.agent_in_room_number)[0] == self.goal_floor

    def _reward_function(self, state):
        if state.q == 1:
            return 100
        elif state.q == 0:
            return -1
        else:
            return -100

    def _floor_number(self, state):
        return self.l1_domain.get_floor_numbers(state.room_number)[0]

class CubeL1State(State):
    def __init__(self, room_number, q, is_terminal=False):
        State.__init__(self, data=[room_number, q], is_terminal=is_terminal)
        self.agent_in_room_number = room_number
        self.q = q # logic state

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return 'Agent in room {}, Q: {}'.format(self.agent_in_room_number, self.q)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, CubeL1State) and self.agent_in_room_number == other.agent_in_room_number

class CubeL1GroundedAction(NonPrimitiveAbstractTask):
    def __init__(self, l1_action_string, subtasks, lowerDomain):
        self.action = l1_action_string
        self.goal_room = self.extract_goal_room(self.action)
        #self.goal_state = CubeL1State(goal_room, 0, is_terminal=True)

        self.goal_constraints = {'goal': 'a', 'stay': '~a'}
        self.ap_maps = {'a': (1, 'state', self.goal_room)}
        tf, rf = self._terminal_function, self._reward_function
        self.l0_domain = lowerDomain
        NonPrimitiveAbstractTask.__init__(self, l1_action_string, subtasks, tf, rf)

    @classmethod
    def extract_goal_room(cls, action):
        room_numbers = re1.findall(r'\d+', action)
        if len(room_numbers) == 0:
            raise ValueError('unable to extract room number from L1Action {}'.format(action))
        return int(room_numbers[0])

    def _terminal_function(self, state):
        return self.l0_domain.get_room_numbers((state.x, state.y, state.z))[0] == self.goal_room


    def _reward_function(self, state):
        if state.q == 1:
            return 100
        elif state.q == 0:
            return -1
        else:
            return -100


    def _room_number(self, state):
        return self.l0_domain.get_room_numbers((int(state.x), int(state.y), int(state.z)))[0] # TODO: Check what l0_domain is

class CubeRootL2GroundedAction(RootTaskNode):
    def __init__(self, action_str, subtasks, l2_domain, terminal_func, reward_func, constraints, ap_maps):
        self.action = 'Root_' + action_str
        self.goal_constraints = constraints
        self.ap_maps = ap_maps

        RootTaskNode.__init__(self, self.action, subtasks, l2_domain, terminal_func, reward_func)

class CubeRootL1GroundedAction(RootTaskNode):
    def __init__(self, action_str, subtasks, l1_domain, terminal_func, reward_func, constraints, ap_maps):
        self.action = 'Root_' + action_str
        #self.goal_state = CubeL1State(CubeL1GroundedAction.extract_goal_room(action_str), is_terminal=True)
        self.goal_constraints = constraints
        self.ap_maps = ap_maps

        RootTaskNode.__init__(self, self.action, subtasks, l1_domain, terminal_func, reward_func)

class CubeL2MDP(MDP):
    ACTIONS = ["toFloor%d" %ii for ii in range(1,4)]
    def __init__(self, starting_floor=1, gamma=0.99, env_file=[], constraints={}, ap_maps={}):
        self.terminal_func = lambda state: state.q != 0
        self.constraints = constraints
        self.ap_maps = ap_maps

        if len(env_file) != 0:
            self.cube_env = env_file[0]
            CubeL2MDP.ACTIONS = self.cube_env['L2ACTIONS']
        else:
            print("Input: env_file")

        initial_state = CubeL2State(starting_floor, self._transition_q(starting_floor, ""))
        if initial_state.q != 0:
            initial_state.set_terminal(True)

        MDP.__init__(self, CubeL2MDP.ACTIONS, self._transition_func, self._reward_func, init_state=initial_state,
                     gamma=gamma)

    def _reward_func(self, state, action):

        next_state = self._transition_func(state, action)

        if next_state.q == 0: # stay
            reward = -1
        elif next_state.q == 1:  # success
            reward = 100
        elif next_state.q == -1:  # fail
            reward = -100

        return reward

    def _transition_func(self, state, action):
        if state.is_terminal():
            return state

        current_floor = state.agent_on_floor_number
        next_state = None
        action_floor_number = int(action.split('toFloor')[1])

        if abs(current_floor - action_floor_number) == 1:  # transition function

            evaluated_APs = self._evaluate_APs(action_floor_number)

            for ap in evaluated_APs.keys():
                exec('%s = symbols(\'%s\')' % (ap, ap))

            if eval(self.constraints['goal']).subs(evaluated_APs):
                next_q = 1
            elif eval(self.constraints['stay']).subs(evaluated_APs):
                next_q = 0
            else:
                next_q = -1

            next_state = CubeL2State(action_floor_number, next_q)
            if next_state.q != 0:
                next_state.set_terminal(True)

        if next_state is None:
            next_state = state


        return next_state

    def _transition_q(self, floor_num, action):
        # evaluate APs
        evaluated_APs = self._evaluate_APs(floor_num)

        # q state transition
        # define symbols
        for ap in evaluated_APs.keys():
            exec('%s = symbols(\'%s\')' % (ap, ap))
        # evaluation
        if eval(self.constraints['goal']).subs(evaluated_APs):  # goal
            next_q = 1
        elif eval(self.constraints['stay']).subs(evaluated_APs):  # keep planning
            next_q = 0
        else:  # fail
            next_q = -1

        return next_q


    def _evaluate_APs(self, floor_num):
        evaluated_APs = {}
        for ap in self.ap_maps.keys():
            if (self.ap_maps[ap][0] == 2) and (self.ap_maps[ap][2] == floor_num):
                evaluated_APs[ap] = True
            else:
                evaluated_APs[ap] = False
        return evaluated_APs

    def __str__(self):
        return 'AbstractCubeL2MDP: InitState: {}, Goal: {}'.format(self.init_state, self.constraints['goal'])

#    @classmethod
    def action_for_floor_number(self, floor_number):
        for action in CubeL2MDP.ACTIONS:
            if str(floor_number) in action:
                return action
        raise ValueError('unable to find action corresponding to floor {}'.format(floor_number))

class CubeL1MDP(MDP):
    ACTIONS = ["toRoom%d" %ii for ii in range(1, 11)]  # actions??
    def __init__(self, starting_room=1, gamma=0.99, slip_prob=0.0, env_file=[], constraints = {}, ap_maps = {}):
        # TODO: work
        self.terminal_func = lambda state: state.q != 0
        self.constraints = constraints
        self.ap_maps = ap_maps
        self.slip_prob = slip_prob

        if len(env_file) != 0:
            self.cube_env = env_file[0]
            CubeL1MDP.ACTIONS = self.cube_env['L1ACTIONS']
        else:
            print("Input: env_file")

        initial_state = CubeL1State(starting_room, self._transition_q(starting_room, ""))
        if initial_state.q != 0:
            initial_state.set_terminal(True)

        MDP.__init__(self, CubeL1MDP.ACTIONS, self._transition_func, self._reward_func, init_state=initial_state,
                     gamma=gamma)

    def _reward_func(self, state, action):

        next_state = self._transition_func(state, action)

        if next_state.q == 0: # stay
            reward = -1
        elif next_state.q == 1:  # success
            reward = 100
        elif next_state.q == -1:  # fail
            reward = -100
        #print(state, action, reward)
        return reward


    def _transition_func(self, state, action):
        if state.is_terminal():
            return state

        current_room = state.agent_in_room_number
        next_state = None
        action_room_number = int(action.split('toRoom')[1])
        if current_room in self.cube_env['transition_table'].keys():
            if action_room_number in self.cube_env['transition_table'][current_room]:

                # evaluation
                evaluated_APs = self._evaluate_APs(action_room_number)

                for ap in evaluated_APs.keys():
                    exec('%s = symbols(\'%s\')' % (ap, ap))

                if eval(self.constraints['goal']).subs(evaluated_APs):
                    next_q = 1
                elif eval(self.constraints['stay']).subs(evaluated_APs):
                    next_q = 0
                else:
                    next_q = -1

                next_state = CubeL1State(action_room_number, next_q)

                if next_state.q != 0:
                    next_state.set_terminal(True)

        if next_state is None:
            next_state = state

            #next_state._is_terminal = (next_state.q == 1)

        return next_state

    def _transition_q(self, room_num, action):
        # evaluate APs
        evaluated_APs = self._evaluate_APs(room_num)

        # q state transition
        # define symbols
        for ap in evaluated_APs.keys():
            exec('%s = symbols(\'%s\')' % (ap, ap))
        # evaluation
        if eval(self.constraints['goal']).subs(evaluated_APs):  # goal
            next_q = 1
        elif eval(self.constraints['stay']).subs(evaluated_APs):  # keep planning
            next_q = 0
        else:  # fail
            next_q = -1

        return next_q

    def _evaluate_APs(self, room_number):
        evaluated_APs = {}

        for ap in self.ap_maps.keys():
            if (self.ap_maps[ap][0] == 1) and (self.ap_maps[ap][2] == room_number):
                evaluated_APs[ap] = True
            elif (self.ap_maps[ap][0] == 2) and (self.ap_maps[ap][2] == self.get_floor_numbers(room_number)[0]):
                evaluated_APs[ap] = True
            else:
                evaluated_APs[ap] = False
        return evaluated_APs


    def get_floor_numbers(self, room_number):
        floor_numbers = []
        for i in range(1,self.cube_env['num_floor']+1):
            if room_number in self.cube_env['floor_to_rooms'][i]:
                floor_numbers.append(i)
        return floor_numbers


    def __str__(self):
        return 'AbstractFourRoomMDP: InitState: {}, GoalState: {}'.format(self.init_state, self.constraints['goal'])

#    @classmethod
    def action_for_room_number(self, room_number):
        for action in CubeL1MDP.ACTIONS:
            if str(room_number) in action:
                return action
        raise ValueError('unable to find action corresponding to room {}'.format(room_number))

# -----------------------------------
# Debug functions
# -----------------------------------

def debug_l1_grid_world():
    def get_l1_policy(start_room=None, goal_room=None, mdp=None):
        if mdp is None:
            mdp = CubeL1MDP(start_room, goal_room)
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

if __name__ == '__main__':
    mdp = CubeL1MDP(env_file=[build_cube_env()])
    print('done')