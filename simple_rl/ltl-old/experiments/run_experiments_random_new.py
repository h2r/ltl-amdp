import time
import os
import random
from simple_rl.ltl.LTLautomataClass import LTLautomata
from simple_rl.ltl.AMDP.RoomCubePlainMDPClass import RoomCubePlainMDP
from simple_rl.ltl.AMDP.LtlAMDPClass import LTLAMDP
from simple_rl.ltl.settings.build_cube_env_2 import build_cube_env
from simple_rl.planning import ValueIteration
from simple_rl.ltl.experiments.run_experiments import run_plain_pMDP, run_aMDP, run_aMDP_lowest

NUM_SIMULATE = 150  # the number of random cases
NUM_SIMULATE = 1  # the number of random cases

flag_verbose = True
flag_save = True
num_env = 2

if __name__ == '__main__':



    print('Inside main!')
    print('num_simulations: ', NUM_SIMULATE)
    print('num_env: ', num_env)
    cube_env = build_cube_env()

    # Define a set of LTL formulas
    formula_set = ['Fa', 'F (a & F b)', 'F(a & F( b & Fc))'] #, '~a U b']
    num_formula = len(formula_set)
    AP_keys = ['a', 'b', 'c', 'd', 'e']
    AP_num = [1, 2, 3, 2]  # the number of atomic propositions in formula_set[i]
    init_loc = (1,1,1)
    init_room = 1
    init_floor = 1

    # Initialize variables for the result
    run_time_amdp = []; run_len_amdp = []; is_correct_amdp = []; backup_amdp = []
    run_time_amdp_lowest = []; run_len_amdp_lowest = []; is_correct_amdp_lowest = []; backup_amdp_lowest=[]
    run_time_plain = []; run_len_plain = [];  is_correct_plain = []; backup_plain =[]

    if flag_save:
        # changed!
        file = open("{}/results/result_time_random_new.txt".format(os.getcwd()), "a")
        file.write("=== Env {} ==============================================\n".format(num_env))
        file.write("Run {} times\n".format(NUM_SIMULATE))
        file.write("time_AMDP, time_AMDP_level0, time_pMDP, len_AMDP, len_AMDP_level0,"
                   " len_pMDP, correct_AMDP, correct_AMDP_level0, correct_pMDP, backup_AMDP, backup_AMDP_level0, backup_pMDP"
                   ", LTL, ap_maps\n")
        file.close()



    print('Starting for loop');
    
    for i in range(0, NUM_SIMULATE):
        # randomly select the type of formula
        ftype = random.randint(0, num_formula-1)
        ltl_formula = formula_set[ftype]

        # randomly define ap_maps
        ap_maps = {}
        for iap in range(0, AP_num[ftype]):

            level = random.randint(0, 2)  # select random level
            while True:

                if level == 0:
                    while True:
                        state = (random.randint(1, cube_env['len_x']), random.randint(1, cube_env['len_y']), random.randint(1, cube_env['len_z']))
                        if (state not in cube_env['walls']):
                            break
                elif level == 1:
                    state = random.randint(1, cube_env['num_room'])
                elif level == 2:
                    state = random.randint(1, cube_env['num_floor'])

                if [level, 'state', state] not in ap_maps.values():
                    break

            ap_maps[AP_keys[iap]] = [level, 'state', state]

        # Run
        print("[Trial {}] {}, {}\n".format(i, ltl_formula, ap_maps))

        # Experiment: AMDP
        print("[Trial {}] AP-MDP ----------------------------------------".format(i))
        t, l, flag_success, _, _, backup_num = run_aMDP(init_loc, ltl_formula, cube_env, ap_maps, verbose=flag_verbose)
        run_time_amdp.append(t)
        run_len_amdp.append(l)
        backup_amdp.append(backup_num)
        is_correct_amdp.append(flag_success)
        print("  [AP-MDP]  Time: {} seconds, the number of actions: {}, backup: {}"
              .format(round(t, 3), l, backup_num))

        # Experiment: decomposed LTL and solve it at the lowest level
        print("[Trial {}] AP-MDP at level 0 ----------------------------------------".format(i))
        t, l, flag_success, _, _, backup_num = run_aMDP_lowest(init_loc, ltl_formula, cube_env, ap_maps, verbose=flag_verbose)
        run_time_amdp_lowest.append(t)
        run_len_amdp_lowest.append(l)
        is_correct_amdp_lowest.append(flag_success)
        backup_amdp_lowest.append(backup_num)
        print("  [AP-MDP at level 0]  Time: {} seconds, the number of actions: {}, backup: {}"
              .format(round(t, 3), l, backup_num))

        # Experiment: Plain MDP
        print("[Trial {}] Plain ----------------------------------------".format(i))
        t, l, flag_success, _, _, backup_num = run_plain_pMDP(init_loc, ltl_formula, cube_env, ap_maps, verbose=flag_verbose)
        run_time_plain.append(t)
        run_len_plain.append(l)
        is_correct_plain.append(flag_success)
        backup_plain.append(backup_num)
        print("  [Plain] Time: {} seconds, the number of actions: {}, backup: {}"
              .format(round(t, 3), l, backup_num))

        if flag_save:
            file = open("{}/results/result_time_random.txt".format(os.getcwd()), "a")
            file.write("{}, {}, {}, ".format(round(run_time_amdp[-1],4), round(run_time_amdp_lowest[-1],4),
                                             round(run_time_plain[-1],4)))  # time
            file.write("{}, {}, {}, ".format(round(run_len_amdp[-1], 3), round(run_len_amdp_lowest[-1], 3),
                                             round(run_len_plain[-1], 3)))  # length
            file.write("{}, {}, {}, ".format(round(is_correct_amdp[-1], 3), round(is_correct_amdp_lowest[-1], 3),
                                             round(is_correct_plain[-1], 3)))  # correct
            file.write("{}, {}, {}, ".format(backup_amdp[-1], backup_amdp_lowest[-1], backup_plain[-1]))
            file.write("{}, {}\n".format(ltl_formula, ap_maps))
            file.close()

    






