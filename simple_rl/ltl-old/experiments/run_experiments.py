import time
import os
from simple_rl.ltl.AMDP.RoomCubePlainMDPClass import RoomCubePlainMDP
from simple_rl.ltl.AMDP.LtlAMDPClass import LTLAMDP
from simple_rl.ltl.settings.build_cube_env_1 import build_cube_env
from simple_rl.planning import ValueIteration

def run_plain_pMDP(init_loc, ltl_formula, cube_env, ap_maps, verbose=False):
    start_time = time.time()
    mdp = RoomCubePlainMDP(init_loc = init_loc, ltl_formula=ltl_formula, env_file=[cube_env],
                           ap_maps=ap_maps)

    value_iter = ValueIteration(mdp, sample_rate=1, max_iterations=50)
    value_iter.run_vi()

    # Value Iteration
    action_seq, state_seq = value_iter.plan(mdp.get_init_state())

    computing_time = time.time() - start_time

    # Print
    if verbose:
        print("=====================================================")
        print("Plain: Plan for ", ltl_formula)
        for i in range(len(action_seq)):
            room_number, floor_number = mdp._get_abstract_number(state_seq[i])

            print(
                "\t {} in room {} on the floor {}, {}".format(state_seq[i], room_number, floor_number, action_seq[i]))
        room_number, floor_number = mdp._get_abstract_number(state_seq[-1])
        print("\t {} in room {} on the floor {}".format(state_seq[-1], room_number, floor_number))

    # success?
    if len(state_seq) <= 1:
        flag_success = -1
    else:
        if mdp.automata.aut_spot.state_is_accepting(state_seq[-1].q):
            flag_success = 1
        else:
            flag_success = 0


    return computing_time, len(action_seq), flag_success, state_seq, action_seq, value_iter.get_num_backups_in_recent_run()

def run_aMDP(init_loc, ltl_formula, cube_env, ap_maps, verbose=False):
    start_time = time.time()
    ltl_amdp = LTLAMDP(ltl_formula, ap_maps, env_file=[cube_env], slip_prob=0.0, verbose=verbose)

    # ltl_amdp.solve_debug()
    sseq, aseq, len_actions, backup_num = ltl_amdp.solve(init_loc)

    computing_time = time.time() - start_time

    # success?
    if len_actions == 0:# or len(sseq) == 0:
        flag_success = -1
        #len_actions = 0
    else:
        if sseq[-1][-1].q == 1:
            flag_success = 1
        else:
            flag_success = 0

    return computing_time, len_actions, flag_success, sseq, aseq, backup_num

def run_aMDP_lowest(init_loc, ltl_formula, cube_env, ap_maps, verbose=False):
    start_time = time.time()
    ltl_amdp = LTLAMDP(ltl_formula, ap_maps, env_file=[cube_env], slip_prob=0.0, verbose=verbose)

    # ltl_amdp.solve_debug()
    sseq, aseq, len_actions, backup_num = ltl_amdp.solve(init_loc, FLAG_LOWEST=True)

    computing_time = time.time() - start_time

    # success?
    if len_actions == 0:
        flag_success = -1
    else:
        if sseq[-1][-1].q == 1:
            flag_success = 1
        else:
            flag_success = 0

    return computing_time, len_actions, flag_success, sseq, aseq, backup_num


if __name__ == '__main__':

    cube_env1 = build_cube_env()

    # define scenarios for a large environment
    formula_set1 = ['Fa', 'F (a & F b)',  'F(a & F( b & Fc))', '~a U b', 'F (a & F b)','F(a & F( b & Fc))']
    ap_maps_set1 = {}
    ap_maps_set1[0] = {'a': [2, 'state', 3]}
    ap_maps_set1[1] = {'a': [0, 'state', (2,4,1)], 'b': [1,'state', 7]}
    ap_maps_set1[2] = {'a': [1, 'state', 9], 'b': [2, 'state', 3], 'c': [1, 'state', 17]}
    ap_maps_set1[3] = {'a': [1, 'state', 2], 'b': [2, 'state', 3]}
    ap_maps_set1[4] = {'a': [1, 'state', 9], 'b': [1, 'state', 17]}
    ap_maps_set1[5] = {'c': [0, 'state', (1, 4, 3)], 'a': [2, 'state', 1], 'b': [2, 'state', 2]}

    # define scenarios for a large environment
    formula_set2 = ['Fa', 'Fa', 'F (a & F b)', '~a U b', 'F(a & F( b & F c))']
    ap_maps_set2 = {}
    ap_maps_set2[0] = {'a': [1, 'state', 8]}
    ap_maps_set2[1] = {'a': [2, 'state', 6]}
    ap_maps_set2[2] = {'a': [2, 'state', 4], 'b': [1, 'state', 6]}
    ap_maps_set2[3] = {'a': [1, 'state', 11], 'b': [1, 'state', 12]}
    ap_maps_set2[4] = {'a': [1, 'state', 5], 'b': [2, 'state', 3], 'c': [0, 'state', (11, 11, 3)]}

    formula_set3 = ['Fa', '~a U b', 'F((a | b) & F c)', 'F (a & F b)']
    ap_maps_set3 = {}
    ap_maps_set3[0] = {'a': [2, 'state', 3]}
    ap_maps_set3[1] = {'a': [1, 'state', 2], 'b': [2, 'state', 3]}
    ap_maps_set3[2] = {'a': [2, 'state', 2], 'b': [1, 'state', 2], 'c': [2, 'state', 1]}
    ap_maps_set3[3] = {'a': [2, 'state', 2], 'b': [1, 'state', 8]}


    # simulation settings
    run_num = 1.0   #the number of run
    flag_verbose = False  # Show result paths
    flag_save = False
    num_env = 1 #environment name : build_cube_env(num_env).py 3: for examples
    init_loc = (1,1,1)
    # select the world (1: small, 2: large cube world)
    formula_set = eval("formula_set{}".format(num_env))
    ap_maps_set = eval("ap_maps_set{}".format(num_env))

    for num_case in [5]:
        print("+++++++++++++++++ Case: {} +++++++++++++++++++".format(num_case))
        if flag_save:
            file = open("{}/results/result_time.txt".format(os.getcwd()), "a")

        ltl_formula = formula_set[num_case]
        ap_maps = ap_maps_set[num_case]

        #initialize
        run_time_plain = 0.0
        run_time_amdp = 0.0
        run_time_amdp_lowest = 0.0
        run_len_plain = 0.0
        run_len_amdp = 0.0
        run_len_amdp_lowest = 0.0

        for i in range(int(run_num)):

            print("* Trial {}".format(i))

            # Experiment: AMDP
            print("[Trial {}] AP-MDP ----------------------------------------".format(i))
            t, l, _, _,_, backup= run_aMDP(init_loc, ltl_formula, cube_env1, ap_maps, verbose=flag_verbose)
            run_time_amdp = run_time_amdp + t
            run_len_amdp = run_len_amdp + l
            print("  [AP-MDP]  Time: {} seconds, the number of actions: {}, backup: {}"
                  .format(round(t, 3), l, backup))

            # Experiment: decomposed LTL and solve it at the lowest level
            print("[Trial {}] AP-MDP at level 0 ----------------------------------------".format(i))
            t, l, _, _,_, backup = run_aMDP_lowest(init_loc, ltl_formula, cube_env1, ap_maps, verbose=flag_verbose)
            run_time_amdp_lowest = run_time_amdp_lowest + t
            run_len_amdp_lowest = run_len_amdp_lowest + l
            print("  [AP-MDP at level 0]  Time: {} seconds, the number of actions: {}, backup: {}"
                  .format(round(t, 3), l, backup))

            # Experiment: Plain MDP

            print("[Trial {}] Plain ----------------------------------------".format(i))
            t, l, _, _,_, backup = run_plain_pMDP(init_loc, ltl_formula, cube_env1, ap_maps, verbose=flag_verbose)
            run_time_plain = run_time_plain + t
            run_len_plain = run_len_plain + l
            print("  [Plain] Time: {} seconds, the number of actions: {}, backup: {}"
                  .format(round(t, 3), l, backup))


        print("* Summary: " + ltl_formula)
        print("  AP-MDP: {}s, {}".format(round(run_time_amdp / run_num, 3), run_len_amdp / run_num))
        print("  AP-MDP at level 0: {}s, {}".format(round(run_time_amdp_lowest / run_num, 3), run_len_amdp_lowest / run_num))
        print("  Product-MDP: {}s, {}".format(round(run_time_plain / run_num, 3), run_len_plain / run_num))

        if flag_save:
            file.write("=== Env {} ==============================================\n".format(num_env))
            file.write("Run {} times\n".format(run_num))
            file.write("Task:\t"+ltl_formula+"\n")
            file.write("AP:\t{}\n".format(ap_maps))
            file.write("AP-MDP:\t{}s, {}\n".format(round(run_time_amdp / run_num, 3), run_len_amdp / run_num))
            file.write("AP-MDP at level 0:\t{}s, {}\n".format(round(run_time_amdp_lowest / run_num, 3),
                                                      run_len_amdp_lowest / run_num))
            file.write("Product-MDP:\t{}s, {}\n".format(round(run_time_plain / run_num, 3), run_len_plain / run_num))

            file.close()




