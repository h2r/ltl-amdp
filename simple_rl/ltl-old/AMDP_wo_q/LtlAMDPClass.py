import sympy
import spot
from simple_rl.ltl.LTLautomataClass import LTLautomata

# Generic AMDP imports.
from simple_rl.amdp.AMDPSolverClass import AMDPAgent
from simple_rl.amdp.AMDPTaskNodesClass import PrimitiveAbstractTask

# Abstract grid world imports.
from simple_rl.ltl.AMDP.RoomCubeMDPClass import RoomCubeMDP
from simple_rl.ltl.AMDP.RoomCubeStateClass import RoomCubeState
from simple_rl.ltl.AMDP.AbstractCubeMDPClass import *
from simple_rl.ltl.AMDP.AbstractCubePolicyGeneratorClass import *
from simple_rl.ltl.AMDP.AbstractCubeStateMapperClass import *

from simple_rl.ltl.settings.build_cube_env_1 import build_cube_env

from simple_rl.run_experiments import run_agents_on_mdp


class LTLAMDP():
    def __init__(self, ltlformula, ap_maps, slip_prob=0.01):
        '''

        :param ltlformula: string, ltl formulation ex) a & b
        :param ap_maps: atomic propositions are denoted by alphabets. It should be mapped into states or actions
                        ex) {a:[(int) level, 'action' or 'state', value], b: [0,'action', 'south']
        '''
        self.automata = LTLautomata(ltlformula) # Translate LTL into the automata
        self.ap_maps = ap_maps
        self.cube_env = build_cube_env() #define environment
        self._generate_AP_tree() # relationship between atomic propositions
        # simplify automata
        self.automata._simplify_dict(self.relation_TF)
        self.slip_prob = slip_prob


    def solve_debug(self):
        constraints={'goal': 'a', 'stay': '~a'}
        sub_ap_maps={'a': (2, 'state', 3), 'b': (1, 'state', 2), 'c': (0, 'state', (1, 4, 1))}

        # 2. Parse: Which level corresponds to the current sub - problem
        sub_level = 2
        for ap in sub_ap_maps.keys():
            if ap in constraints['goal'] or ap in constraints['stay']:

                sub_level = min(sub_level, sub_ap_maps[ap][0])

        # 3. Solve AMDP

        if sub_level == 0:
            self._solve_subproblem_L0(constraints=constraints, ap_maps=sub_ap_maps)

        elif sub_level == 1:
            # solve
            self._solve_subproblem_L1(constraints=constraints, ap_maps=sub_ap_maps)
        elif sub_level == 2:
            # solve
            self._solve_subproblem_L2(constraints=constraints, ap_maps=sub_ap_maps)

    def solve(self, init_loc=(1, 1, 1)):
        Q_init = self.automata.init_state
        Q_goal = self.automata.get_accepting_states()

        [q_paths, q_words]=self.automata.findpath(Q_init, Q_goal[0])   # Find a path of states of automata

        n_path = len(q_paths) # the number of paths

        # Find a path in the environment
        for np in range(0, n_path):

            cur_path = q_paths[np] # current q path
            cur_words = q_words[np] # current q words
            cur_loc = init_loc

            action_seq = []
            state_seq = []

            for tt in range(0, len(cur_words)):
                trans_fcn = self.automata.trans_dict[cur_path[tt]]
                # 1. extract constraints
                constraints = {}
                constraints['goal'] = cur_words[tt]
                constraints['stay'] = [s for s in trans_fcn.keys() if trans_fcn[s] == cur_path[tt]][0]

                # 2. Parse: Which level corresponds to the current sub - problem
                sub_ap_maps = {}
                sub_level = 2
                for ap in self.ap_maps.keys():
                    if ap in constraints['goal'] or ap in constraints['stay']:
                        sub_ap_maps[ap] = ap_maps[ap]
                        sub_level = min(sub_level, sub_ap_maps[ap][0])
                print("----- Solve in level {} MDP -----".format(sub_level))
                # 3. Solve AMDP
                if sub_level == 0:
                    action_seq_sub, state_seq_sub = self._solve_subproblem_L0(init_locs=cur_loc, constraints=constraints, ap_maps =sub_ap_maps)

                elif sub_level == 1:
                    # solve
                    action_seq_sub, state_seq_sub = self._solve_subproblem_L1(init_locs=cur_loc, constraints=constraints, ap_maps=sub_ap_maps)
                elif sub_level == 2:
                    # solve
                    action_seq_sub, state_seq_sub = self._solve_subproblem_L2(init_locs=cur_loc, constraints=constraints, ap_maps=sub_ap_maps)

                # update
                state_seq.append(state_seq_sub)
                action_seq.append(action_seq_sub)
                cur_loc = (state_seq_sub[-1].x, state_seq_sub[-1].y, state_seq_sub[-1].z)

            print("=====================================================")
            print("Plan for a path {} in DBA".format(np))
            for k in range(len(action_seq)):
                for i in range(len(action_seq[k])):
                    room_number, floor_number = self._get_abstract_number(state_seq[k][i])

                    print("\t {} in room {} on the floor {}, {}".format(state_seq[k][i], room_number, floor_number, action_seq[k][i]))
                print('\t----------------------------------------')
            room_number, floor_number = self._get_abstract_number(state_seq[k][-1])
            print("\t {} in room {} on the floor {}".format(state_seq[k][-1], room_number, floor_number))

            print("=====================================================")
    def _get_room_number(self, state):
        room_number = 0
        for r in range(1, self.cube_env['num_room'] + 1):
            if (state.x, state.y, state.z) in self.cube_env['room_to_locs'][r]:
                room_number = r

        return room_number

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


    def _solve_subproblem_L0(self, init_locs=(1, 1, 1), constraints={}, ap_maps={}): #TODO
        mdp = RoomCubeMDP(init_loc=init_locs, env_file = [self.cube_env], constraints = constraints, ap_maps = ap_maps, slip_prob=self.slip_prob)
        value_iter = ValueIteration(mdp, sample_rate = 5)
        value_iter.run_vi()

        # Value Iteration.
        action_seq, state_seq = value_iter.plan(mdp.get_init_state())

        # TODO: Extract policy by value_iter.policy(state)... What about returning value_iter?
        print("Plan for", mdp)
        for i in range(len(action_seq)):
            print("\t", state_seq[i], action_seq[i])
        print("\t", state_seq[-1])

        return action_seq, state_seq


    def _solve_subproblem_L1(self, init_locs=(1, 1, 1), constraints={}, ap_maps={}):

        # define l0 domain
        l0Domain = RoomCubeMDP(init_loc=init_locs, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps,
                               slip_prob=self.slip_prob)

        # define l1 domain
        start_room = l0Domain.get_room_numbers(init_locs)
        l1Domain = CubeL1MDP(start_room, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps,
                             slip_prob=self.slip_prob)

        policy_generators = []
        l0_policy_generator = CubeL0PolicyGenerator(l0Domain, env_file=[self.cube_env])
        l1_policy_generator = CubeL1PolicyGenerator(l0Domain, AbstractCubeL1StateMapper(l0Domain), env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps)

        policy_generators.append(l0_policy_generator)
        policy_generators.append(l1_policy_generator)

        # 2 levels
        l1Subtasks = [PrimitiveAbstractTask(action) for action in l0Domain.ACTIONS]
        a2rt = [CubeL1GroundedAction(a, l1Subtasks, l0Domain) for a in l1Domain.ACTIONS]
        l1Root = CubeRootL1GroundedAction(constraints['goal'], a2rt, l1Domain,
                                          l1Domain.terminal_func, l1Domain.reward_func, constraints=constraints, ap_maps=ap_maps)

        agent = AMDPAgent(l1Root, policy_generators, l0Domain)
        agent.solve()

        state = RoomCubeState(init_locs[0], init_locs[1], init_locs[2], 0)
        action_seq = []
        state_seq = [state]
        while state in agent.policy_stack[0].keys():
            action = agent.policy_stack[0][state]
            state = l0Domain._transition_func(state, action)

            action_seq.append(action)
            state_seq.append(state)

        print("Plan")
        for i in range(len(action_seq)):
            print("\t", state_seq[i], action_seq[i])
        print("\t", state_seq[-1])
        return action_seq, state_seq

    def _solve_subproblem_L2(self, init_locs=(1, 1, 1), constraints={}, ap_maps={}):
        # define l0 domain
        l0Domain = RoomCubeMDP(init_loc=init_locs, env_file=[self.cube_env], constraints=constraints,
                               ap_maps=ap_maps, slip_prob= self.slip_prob)

        # define l1 domain
        start_room = l0Domain.get_room_numbers(init_locs)[0]
        start_floor = l0Domain.get_floor_numbers(init_locs)[0]

        l1Domain = CubeL1MDP(start_room, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps)
        l2Domain = CubeL2MDP(start_floor, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps)

        policy_generators = []
        l0_policy_generator = CubeL0PolicyGenerator(l0Domain, env_file=[self.cube_env])
        l1_policy_generator = CubeL1PolicyGenerator(l0Domain, AbstractCubeL1StateMapper(l0Domain),
                                                    env_file=[self.cube_env], constraints=constraints,
                                                    ap_maps=ap_maps)
        l2_policy_generator = CubeL2PolicyGenerator(l1Domain, AbstractCubeL2StateMapper(l1Domain),
                                                    env_file=[self.cube_env], constraints=constraints,
                                                    ap_maps=ap_maps)

        policy_generators.append(l0_policy_generator)
        policy_generators.append(l1_policy_generator)
        policy_generators.append(l2_policy_generator)

        # 2 levels
        l1Subtasks = [PrimitiveAbstractTask(action) for action in l0Domain.ACTIONS]
        a2rt = [CubeL1GroundedAction(a, l1Subtasks, l0Domain) for a in l1Domain.ACTIONS]
        a2rt2 = [CubeL2GroundedAction(a, a2rt, l1Domain) for a in l2Domain.ACTIONS]

        l2Root = CubeRootL2GroundedAction(l2Domain.action_for_floor_number(1), a2rt2, l2Domain,
                                          l2Domain.terminal_func, l2Domain.reward_func, constraints=constraints,
                                          ap_maps=ap_maps)

        agent = AMDPAgent(l2Root, policy_generators, l0Domain)

        # Test - base, l1 domain
        l2Subtasks = [PrimitiveAbstractTask(action) for action in l1Domain.ACTIONS]

        agent.solve()

        # Extract action seq, state_seq
        state = RoomCubeState(init_locs[0], init_locs[1], init_locs[2], 0)
        action_seq = []
        state_seq = [state]
        while state in agent.policy_stack[0].keys():
            action = agent.policy_stack[0][state]
            state = l0Domain._transition_func(state, action)

            action_seq.append(action)
            state_seq.append(state)

        print("Plan")
        for i in range(len(action_seq)):
            print("\t", state_seq[i], action_seq[i])
        print("\t", state_seq[-1])
        return action_seq, state_seq


    def _generate_AP_tree(self): # return the relationship between atomic propositions
        # TODO: WRONG CHECK!
        relation_TF = {}
        for key in self.ap_maps.keys():
            level = ap_maps[key][0]  # current level
            lower_list = []
            notlower_list = []
            samelevel_list = []
            higher_list = []
            nothigher_list = []

            ap = self.ap_maps[key]

            if level == 0:   # the current level
                for key2 in self.ap_maps.keys():
                    ap2 = self.ap_maps[key2]
                    if ap2[0] == 0:  # level 0
                        samelevel_list.append(key2)
                    if ap2[0] == 1:  # level 1
                        if ap2[1] == 'state' and ap[2] in self.cube_env['room_to_locs'][ap2[2]]:
                            higher_list.append(key2)
                        else:
                            nothigher_list.append(key2)
                    if ap2[0] == 2:  # level 2
                        if ap2[1] == 'state' and ap[2] in self.cube_env['floor_to_locs'][ap2[2]]:
                            higher_list.append(key2)
                        else:
                            nothigher_list.append(key2)

            if level == 1:
                for key2 in self.ap_maps.keys():
                    ap2 = self.ap_maps[key2]
                    if ap2[0] == 0 and ap2[1] == 'state':  # lower
                        if ap2[2] in self.cube_env['room_to_locs'][ap[2]]:
                            lower_list.append(key2)
                        else:
                            notlower_list.append(key2)

                    if self.ap_maps[key2][0] == 1: # same level
                        samelevel_list.append(key2)

                    if ap2[0] == 2 and ap2[1] == 'state': # higher level
                        if ap[2] in self.cube_env['floor_to_rooms'][ap2[2]]:
                            higher_list.append(key2)
                        else:
                            nothigher_list.append(key2)

            if level == 2:
                for key2 in self.ap_maps.keys():
                    ap2 = self.ap_maps[key2]
                    if ap2[0] == 0 and ap2[1] == 'state':  # lower
                        if ap2[2] in self.cube_env['floor_to_locs'][ap[2]]:
                            lower_list.append(key2)
                        else:
                            notlower_list.append(key2)

                    if ap2[0] == 1 and ap2[2] in self.cube_env['floor_to_rooms'][self.ap_maps[key][2]]:
                        lower_list.append(key2)
                    elif self.ap_maps[key2][0] == 1:
                        notlower_list.append(key2)

                    if ap2[0] == 2:
                        samelevel_list.append(key2)

            relation_TF[key] = {'lower': lower_list, 'same': samelevel_list, 'lower_not': notlower_list,
                                'higher': higher_list, 'higher_not': nothigher_list}

        self.relation_TF = relation_TF




if __name__ == '__main__':
    ltl_formula = 'F (b & (F a))'
#    ltl_formula = 'F (a&b)'
    ap_maps = {'a': [1, 'state', 7], 'b': [2, 'state', 2], 'c': [2, 'state', 1], 'd': [0, 'state', (6, 1, 1)], 'e': [2, 'state', 1],
               'f': [2, 'state', 2], 'g': [0, 'state', (1, 4, 3)]}
    ltl_amdp = LTLAMDP(ltl_formula, ap_maps, slip_prob=0.0)

    ltl_amdp._generate_AP_tree()
    #ltl_amdp.solve_debug()
    ltl_amdp.solve()
    print("End")












