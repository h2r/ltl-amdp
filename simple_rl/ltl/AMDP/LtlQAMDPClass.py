import sympy
import spot
import time
from simple_rl.ltl.LTLautomataClass import LTLautomata

# Generic AMDP imports.
from simple_rl.ltl.AMDP.AMDPSolverQClass import AMDPAgent
from simple_rl.amdp.AMDPTaskNodesClass import PrimitiveAbstractTask

# Abstract grid world imports.
from simple_rl.ltl.AMDP.RoomCubeMDPClass import RoomCubeMDP
from simple_rl.ltl.AMDP.RoomCubeStateClass import RoomCubeState
from simple_rl.ltl.AMDP.AbstractCubeQMDPClass import *
from simple_rl.ltl.AMDP.AbstractCubeQPolicyGeneratorClass import *
from simple_rl.ltl.AMDP.AbstractCubeQStateMapperClass import *

from simple_rl.ltl.settings.build_cube_env_1 import build_cube_env

from simple_rl.run_experiments import run_agents_on_mdp


class LTLAMDP():
    def __init__(self, ltlformula, ap_maps, env_file=[], slip_prob=0.01, verbose=False):
        '''

        :param ltlformula: string, ltl formulation ex) a & b
        :param ap_maps: atomic propositions are denoted by alphabets. It should be mapped into states or actions
                        ex) {a:[(int) level, 'action' or 'state', value], b: [0,'action', 'south']
        '''
        self.automata = LTLautomata(ltlformula) # Translate LTL into the automata
        self.ap_maps = ap_maps
        self.cube_env = env_file[0] #build_cube_env() #define environment
        self._generate_AP_tree() # relationship between atomic propositions
        # simplify automata
        self.automata._simplify_dict(self.relation_TF)
        self.slip_prob = slip_prob
        self.verbose = verbose

    def solve(self, init_loc=(1, 1, 1), FLAG_LOWEST=False):
        Q_init = self.automata.init_state
        Q_goal = self.automata.get_accepting_states()
        init_state = RoomCubeState(init_loc[0], init_loc[1], init_loc[2], Q_init)

        backup_num = 0

        [q_paths, q_words]=self.automata.findpath(Q_init, Q_goal[0])   # Find a path of states of automata

        n_path = len(q_paths) # the number of paths

        len_action_opt = 1000
        state_seq_opt = []
        action_seq_opt = []
        # Find a path in the environment
        for np in range(0, n_path):
            flag_success = True
            cur_path = q_paths[np] # current q path
            cur_words = q_words[np] # current q words
            cur_loc = init_loc

            action_seq = []
            state_seq = []
            cur_stay= []

            len_action = 0

            cur_state = init_state
            for tt in range(0, len(cur_words)):
                trans_fcn = self.automata.trans_dict[cur_path[tt]]
                # 1. extract constraints

                constraints= {'Qg': [cur_path[tt+1]], 'Qs': [cur_path[tt]], 'Sg':[], 'Ss': [], 'mode': 'root'} #TODO

                constraints2 = {}
                constraints2['goal'] = cur_words[tt]
                constraints2['stay'] = [s for s in trans_fcn.keys() if trans_fcn[s] == cur_path[tt]][0]
                #cur_stay.append(constraints['stay'])
                # 2. Parse: Which level corresponds to the current sub - problem
                sub_ap_maps = {}
                sub_level = 2
                for ap in self.ap_maps.keys():
                    if ap in constraints2['goal'] or ap in constraints2['stay']:
                        sub_ap_maps[ap] = self.ap_maps[ap]
                        sub_level = min(sub_level, sub_ap_maps[ap][0])
                # solve at the lowest level
                if FLAG_LOWEST:
                    sub_level = 0

                if self.verbose:
                    print("----- Solve in level {} MDP : goal {}, stay {} -----".format(sub_level, constraints['Qg'], constraints['Qs']))
                # 3. Solve AMDP
                if sub_level == 0:
                    action_seq_sub, state_seq_sub, backup_num_sub = self._solve_subproblem_L0(init_locs=cur_loc,
                                                                                              constraints=constraints, ap_maps =sub_ap_maps,
                                                                                              init_state=cur_state)

                elif sub_level == 1:
                    # solve
                    action_seq_sub, state_seq_sub, backup_num_sub = self._solve_subproblem_L1(init_locs=cur_loc, constraints=constraints, ap_maps=sub_ap_maps,
                                                                                              init_state=cur_state)
                elif sub_level == 2:
                    # solve
                    action_seq_sub, state_seq_sub, backup_num_sub = self._solve_subproblem_L2(init_locs=cur_loc, constraints=constraints, ap_maps=sub_ap_maps,
                                                                                              init_state=cur_state)

                # update
                backup_num = backup_num + backup_num_sub
                state_seq.append(state_seq_sub)
                action_seq.append(action_seq_sub)
                cur_loc = (state_seq_sub[-1].x, state_seq_sub[-1].y, state_seq_sub[-1].z)
                cur_state = RoomCubeState(state_seq_sub[-1].x, state_seq_sub[-1].y, state_seq_sub[-1].z, state_seq_sub[-1].q)
                len_action = len_action + len(action_seq_sub)

                if state_seq_sub[-1].q not in constraints['Qg']:
                    flag_success = False
                    break

            if flag_success:
                if len_action_opt > len_action:
                    state_seq_opt = state_seq
                    action_seq_opt = action_seq
                    len_action_opt = len_action
                if self.verbose:
                    print("=====================================================")
                    if flag_success:
                        print("[Success] Plan for a path {} in DBA".format(np))
                    else:
                        print("[Fail] Plan for a path {} in DBA".format(np))

                    for k in range(len(action_seq)):
                        print("Goal: {}, Stay: {}".format(constraints['Qg'], constraints['Qs']))
                        for i in range(len(action_seq[k])):
                            room_number, floor_number = self._get_abstract_number(state_seq[k][i])

                            print("\t {} in room {} on the floor {}, {}".format(state_seq[k][i], room_number, floor_number, action_seq[k][i]))
                        print('\t----------------------------------------')
                    room_number, floor_number = self._get_abstract_number(state_seq[k][-1])
                    print("\t {} in room {} on the floor {}".format(state_seq[k][-1], room_number, floor_number))

                    print("=====================================================")

        return state_seq_opt, action_seq_opt, len_action_opt, backup_num

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


    def _solve_subproblem_L0(self, init_locs=(1, 1, 1), constraints={},
                             ap_maps={}, init_state=[], verbose=False): #TODO
        mdp = RoomCubeMDP(init_loc=init_locs, env_file = [self.cube_env], constraints=constraints, ap_maps = ap_maps,
                          slip_prob=self.slip_prob, automata=self.automata, init_state=init_state)
        value_iter = ValueIteration(mdp, sample_rate = 1)
        value_iter.run_vi()
        num_backup = value_iter.get_num_backups_in_recent_run()

        # Value Iteration.
        action_seq, state_seq = value_iter.plan(mdp.get_init_state())

        if verbose:
            print("Plan for", mdp)
            for i in range(len(action_seq)):
                print("\t", state_seq[i], action_seq[i])
            print("\t", state_seq[-1])

        return action_seq, state_seq, num_backup


    def _solve_subproblem_L1(self, init_locs=(1, 1, 1), constraints={}, ap_maps={}, init_state=[],
                             verbose=False):

        # define l0 domain
        l0Domain = RoomCubeMDP(init_loc=init_locs, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps,
                               slip_prob=self.slip_prob, automata=self.automata, init_state=init_state)
        backup_num = 0
        # if the current state satisfies the constraint already, we don't have to solve it.
        if l0Domain.init_state.q in constraints['Qg']:
            action_seq = []
            state_seq = [l0Domain.init_state]
        else:
            # define l1 domain
            start_room = l0Domain.get_room_numbers(init_locs)[0]
            init_state_L1 = CubeL1State(start_room, init_state.q)
            l1Domain = CubeL1MDP(start_room, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps,
                                 slip_prob=self.slip_prob, automata=self.automata, init_state=init_state_L1)

            policy_generators = []
            l0_policy_generator = CubeL0PolicyGenerator(l0Domain, env_file=[self.cube_env])
            l1_policy_generator = CubeL1PolicyGenerator(l0Domain, AbstractCubeL1StateMapper(l0Domain), env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps)

            policy_generators.append(l0_policy_generator)
            policy_generators.append(l1_policy_generator)

            # 2 levels
            l1Subtasks = [PrimitiveAbstractTask(action) for action in l0Domain.ACTIONS]
            a2rt = [CubeL1GroundedAction(a, l1Subtasks, l0Domain) for a in l1Domain.ACTIONS]
            l1Root = CubeRootL1GroundedAction(l1Domain.action_for_room_number(0), a2rt, l1Domain,
                                              l1Domain.terminal_func, l1Domain.reward_func, constraints=constraints, ap_maps=ap_maps)

            agent = AMDPAgent(l1Root, policy_generators, l0Domain)
            agent.solve()
            backup_num = agent.backup_num

            state = RoomCubeState(init_locs[0], init_locs[1], init_locs[2], 0)
            action_seq = []
            state_seq = [state]
            while state in agent.policy_stack[0].keys():
                action = agent.policy_stack[0][state]
                state = l0Domain._transition_func(state, action)

                action_seq.append(action)
                state_seq.append(state)



        if verbose:
            print("Plan")
            for i in range(len(action_seq)):
                print("\t", state_seq[i], action_seq[i])
            print("\t", state_seq[-1])

        return action_seq, state_seq, backup_num

    def _solve_subproblem_L2(self, init_locs=(1, 1, 1), constraints={},
                             ap_maps={}, verbose=False, init_state=[]):
        # define l0 domain
        l0Domain = RoomCubeMDP(init_loc=init_locs, env_file=[self.cube_env], constraints=constraints,
                               ap_maps=ap_maps, slip_prob= self.slip_prob, automata=self.automata, init_state=init_state)
        backup_num = 0
        # if the current state satisfies the constraint already, we don't have to solve it.
        if l0Domain.init_state.q in constraints['Qg']:
            action_seq = []
            state_seq = [l0Domain.init_state]
        else:
            # define l1 domain
            start_room = l0Domain.get_room_numbers(init_locs)[0]
            start_floor = l0Domain.get_floor_numbers(init_locs)[0]

            init_state_L1 = CubeL1State(start_room, init_state.q)
            init_state_L2 = CubeL2State(start_floor, init_state.q)

            l1Domain = CubeL1MDP(start_room, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps,
                                 automata=self.automata, init_state=init_state_L1)
            l2Domain = CubeL2MDP(start_floor, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps,
                                 automata=self.automata, init_state=init_state_L2)

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
            backup_num = agent.backup_num

            # Extract action seq, state_seq
            state = init_state
            action_seq = []
            state_seq = [state]
            while state in agent.policy_stack[0].keys():
                action = agent.policy_stack[0][state]
                state = l0Domain._transition_func(state, action)

                action_seq.append(action)
                state_seq.append(state)


        # Debuging
        if verbose:
            print("Plan")
            for i in range(len(action_seq)):
                print("\t", state_seq[i], action_seq[i])
            print("\t", state_seq[-1])

        return action_seq, state_seq, backup_num


    def _generate_AP_tree(self): # return the relationship between atomic propositions
        # TODO: WRONG CHECK!
        relation_TF = {}
        for key in self.ap_maps.keys():
            level = self.ap_maps[key][0]  # current level
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

    cube_env = build_cube_env()

    init_loc = (1,1,1)
    ltl_formula = 'F (a & F (b & Fc))'
    ap_maps = {'c': [0, 'state', (1, 4, 3)], 'a': [2, 'state', 1], 'b': [2, 'state', 2]}
    #ltl_formula = 'F(b & Fa)'
    #ap_maps = {'a': [2, 'state', 2], 'b': [1, 'state', 11]}
    #, 'c': [2, 'state', 1], 'd': [0, 'state', (6, 1, 1)], 'e': [2, 'state', 1],
    #           'f': [2, 'state', 3], 'g': [0, 'state', (1, 4, 3)]}
    start_time = time.time()
    ltl_amdp = LTLAMDP(ltl_formula, ap_maps, env_file=[cube_env], slip_prob=0.0, verbose=True)
    #ltl_amdp.automata.findpath_simple(sq=ltl_amdp.automata.init_state, gq=ltl_amdp.automata.get_accepting_states(), ap_maps=ap_maps)

    sseq, aseq, len_actions, backup = ltl_amdp.solve(init_loc, FLAG_LOWEST=False)

    computing_time = time.time()-start_time

    print("Summary")
    print("\t Time: {} seconds, the number of actions: {}, backup: {}"
          .format(round(computing_time, 3), len_actions, backup))














