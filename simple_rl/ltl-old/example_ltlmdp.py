#!/usr/bin/env python

# Python imports.
from __future__ import print_function

# Other imports.
#import srl_example_setup
from simple_rl.ltl.LTLGridWorldMDPClass import LTLGridWorldMDP
from simple_rl.planning import ValueIteration
from simple_rl.agents import QLearningAgent
import spot
spot.setup()
def main():
    ap_map = {'a': (2,2),'b': (6,3), 'c': (5,3), 'd': (4,2)}
    print('Automic propositions, ', ap_map)
    ltlformula = 'F (b & Fa)'
    print('LTL Formula, ', ltlformula)
    # Setup MDP, Agents.
    print('translatinggg')
    a = spot.translate('(a U b) & GFc & GFd', 'BA', 'complete');
    a.show("v"
           "")


    return
    mdp = LTLGridWorldMDP(ltltask=ltlformula, ap_map=ap_map, width=6, height=6, goal_locs=[(6, 6)], slip_prob=0.2)

    mdp.automata.subproblem_flag = 0
    mdp.automata.subproblem_stay = 1
    mdp.automata.subproblem_goal = 0
    value_iter = ValueIteration(mdp, sample_rate=5)
    value_iter.run_vi()

    # Value Iteration.
    print('Value iteration')
    action_seq, state_seq = value_iter.plan(mdp.get_init_state())

    print("Plan for", mdp)
    for i in range(len(action_seq)):
        print("\t", action_seq[i], state_seq[i])

if __name__ == "__main__":
    main()
