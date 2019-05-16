''' FourRoomMDPClass.py: Contains the FourRoom class. '''

# Python imports.
import math
import os
from collections import defaultdict
import numpy as np
import time
from simple_rl.ltl.LTLautomataClass import LTLautomata

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.ltl.AMDP.CubeMDPClass import CubeMDP
from simple_rl.ltl.AMDP.RoomCubeStateClass import RoomCubeState
from simple_rl.planning import ValueIteration

from simple_rl.ltl.AMDP.CubeStateClass import CubeState
from simple_rl.ltl.settings.build_cube_env_1 import build_cube_env

from sympy import *

class RoomCubePlainMDP(CubeMDP):
    ''' Class for a Cube World with Rooms '''

    def __init__(self, ltl_formula= 'Fa', len_x=9, len_y=9, len_z=5, init_loc=(1,1,1),
                 goal_locs=[(9,9,3)], env_file = [],
                 gamma=0.99, slip_prob=0.00, name="cube_room",
                 is_goal_terminal=True, rand_init=False,
                 step_cost=0.0, constraints={'goal':[],'stay':[]}, ap_maps = {}):
        '''
        Args:
            len_x, len_y, len_z (int)
            init_loc (tuple: (int, int,int))
            goal_locs (list of tuples: [(int, int,int)...]
            env_file: specify environment)
            constraints: logic formula of 'goal' and 'stay' for the reward function
                        - goal (large positive), stay (zero), otherwise (large negative)
            ap_maps: dictionary {ap_symbol: (category, state), ...} ex) {a: ('r', [1]), b:('a',west)}
                    category: floor(f), room(r), lowest level action(a), grid cells (c)
        '''

        # Load environment file

        if len(env_file)==0:
            print('Fail to initialize RoomCubeMDP')

        else:
            cube_env = env_file[0]
            len_x = cube_env['len_x']
            len_y = cube_env['len_y']
            len_z = cube_env['len_z']
            walls = cube_env['walls']
            self.num_room = cube_env['num_room']
            self.num_floor = cube_env['num_floor']
            self.room_to_locs = cube_env['room_to_locs']
            self.floor_to_rooms = cube_env['floor_to_rooms']
            self.floor_to_locs = cube_env['floor_to_locs']
            self.cube_env = cube_env

        CubeMDP.__init__(self, len_x, len_y, len_z, init_loc,
                         goal_locs=goal_locs, walls=walls,
                         gamma=gamma, slip_prob=slip_prob, name=name,
                         is_goal_terminal=is_goal_terminal, rand_init=rand_init, step_cost=step_cost)

        self.constraints = constraints  # constraints for LTL
        self.ap_maps = ap_maps  # AP --> real world
        self.automata = LTLautomata(ltl_formula)
        self.init_q = self.automata.init_state

        init_state = RoomCubeState(init_loc[0],init_loc[1],init_loc[2], self.init_q)
        init_state = self.transition_func(init_state, "")
        if self.automata.aut_spot.state_is_accepting(init_state.q):
            init_state.set_terminal(True)

        MDP.__init__(self, RoomCubePlainMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state,
                     gamma=gamma)


    def _transition_func(self, state, action):
        next_state_xyz = super()._transition_func(state, action)

        evaluated_APs = self._evaluate_APs((next_state_xyz.x, next_state_xyz.y, next_state_xyz.z), action)
        next_q = self.automata.transition_func(state.q, evaluated_APs)

        next_state = RoomCubeState(next_state_xyz.x, next_state_xyz.y, next_state_xyz.z, next_q)

        if self.automata.aut_spot.state_is_accepting(next_q):
            next_state.set_terminal(True)

        return next_state

    def is_loc_in_room(self, loc, room_number):
        return loc in self.room_to_locs[room_number]


    def is_loc_on_floor(self, loc, floor_number):
        return loc in self.floor_to_locs[floor_number]

    def get_room_numbers(self, loc):
        room_numbers = []
        for i in range(1, self.num_room+1):
            if loc in self.room_to_locs[i]:
                room_numbers.append(i)
        return room_numbers

    def get_floor_numbers(self, loc):
        room_number = self.get_room_numbers(loc)[0]
        floor_numbers = []
        for i in range(1, self.num_floor+1):
            if room_number in self.floor_to_rooms[i]:
                floor_numbers.append(i)
        return floor_numbers

    def _reward_func(self, state, action): # TODO: Complete
        next_state = self._transition_func(state, action)
        return self.automata.reward_func(next_state.q)

    def _evaluate_APs(self, loc, action): # TODO: Complete
        evaluated_APs ={}

        for ap in self.ap_maps.keys():
            if (self.ap_maps[ap][0] == 0) and (self.ap_maps[ap][1] == 'state'): # level 0
                evaluated_APs[ap] = (loc[0] == self.ap_maps[ap][2][0]) & (loc[1] == self.ap_maps[ap][2][1]) & (loc[2] == self.ap_maps[ap][2][2])

            elif (self.ap_maps[ap][0] == 0 ) and (self.ap_maps[ap][1] == 'action'):
                evaluated_APs[ap] = self.ap_maps[ap][2] in action

            elif self.ap_maps[ap][0] == 1 and (self.ap_maps[ap][1] == 'state'):  # level 1
                evaluated_APs[ap] = self.is_loc_in_room(loc, self.ap_maps[ap][2])

            elif self.ap_maps[ap][0] == 1 and (self.ap_maps[ap][1] == 'action'):  # level 1
                evaluated_APs[ap] = self.ap_maps[ap][2] in action

            elif self.ap_maps[ap][0] == 2 and (self.ap_maps[ap][1] == 'state'):  # level 2
                evaluated_APs[ap] = self.is_loc_on_floor(loc, self.ap_maps[ap][2])

            elif self.ap_maps[ap][0] == 2 and (self.ap_maps[ap][1] == 'action'):  # level 2
                evaluated_APs[ap] = self.ap_maps[ap][2] in action

        return evaluated_APs

    def _get_abstract_number(self, state):
        room_number = 0
        floor_number = 0
        for r in range(1, self.cube_env['num_room'] + 1):
            if (state.x, state.y, state.z) in self.cube_env['room_to_locs'][r]:
                room_number = r
                break

        for f in range(1, self.cube_env['num_floor'] + 1):
            if room_number in self.cube_env['floor_to_rooms'][f]:
                floor_number = f
                break

        return room_number, floor_number


if __name__ == '__main__':
    cube_env1 = build_cube_env()
    ltl_formula = 'F( b & F a)'
    #    ltl_formula = 'F (a&b)'
    ap_maps = {'a': [1, 'state', 7], 'b': [2, 'state', 3]}
    #'c': [1, 'state', 3], 'd': [0, 'state', (6, 1, 1)],
    #           'e': [2, 'state', 1],
    #           'f': [2, 'state', 3], 'g': [0, 'state', (3, 4, 1)]}

    start_time = time.time()
    mdp = RoomCubePlainMDP(ltl_formula=ltl_formula, env_file=[cube_env1],
                           ap_maps = ap_maps)

    value_iter = ValueIteration(mdp, sample_rate=1)
    value_iter.run_vi()

    # Value Iteration
    action_seq, state_seq = value_iter.plan(mdp.get_init_state())

    computing_time = time.time() - start_time

    # print
    print("Plan for", mdp)
    for i in range(len(action_seq)):
        room_number, floor_number = mdp._get_abstract_number(state_seq[i])

        print(
            "\t {} in room {} on the floor {}, {}".format(state_seq[i], room_number, floor_number, action_seq[i]))
    room_number, floor_number = mdp._get_abstract_number(state_seq[-1])
    print("\t {} in room {} on the floor {}".format(state_seq[-1], room_number, floor_number))

    print("Summary")
    print("\t Time: {} seconds, The number of actions: {}"
          .format(round(computing_time, 3), len(action_seq)))
