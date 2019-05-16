#!/usr/bin/env python

# Python imports.
from __future__ import print_function

# Other imports.
#import srl_example_setup
from simple_rl.ltl.LTLGridWorldMDPClass import LTLGridWorldMDP
from simple_rl.planning import ValueIteration
from simple_rl.agents import QLearningAgent

def main():
    ap_map = {'a': (2,2),'b': (6,3), 'c': (5,3), 'd': (4,2)}
    ltlformula = 'F (b & Fa)'
    # Setup MDP, Agents.
    mdp = LTLGridWorldMDP(ltltask=ltlformula, ap_map=ap_map, width=6, height=6, goal_locs=[(6, 6)], slip_prob=0.2)
    mdp.automata.subproblem_flag = 0
    mdp.automata.subproblem_stay = 1
    mdp.automata.subproblem_goal = 0
    value_iter = ValueIteration(mdp, sample_rate=5)
    value_iter.run_vi()

    # Value Iteration.
    action_seq, state_seq = value_iter.plan(mdp.get_init_state())

    print("Plan for", mdp)
    for i in range(len(action_seq)):
        print("\t", action_seq[i], state_seq[i])

def ltl_visualiser(model):
    print('Inside LTL visualiser!')


    from simple_rl.tasks.ltl_amdp.cleanup_block import CleanUpBlock
    from simple_rl.tasks.ltl_amdp.cleanup_door import CleanUpDoor
    from simple_rl.tasks.ltl_amdp.cleanup_room import CleanUpRoom
    from simple_rl.tasks.ltl_amdp.cleanup_task import CleanUpTask
    from simple_rl.tasks.ltl_amdp.CleanupMDPClass import CleanUpMDP

    task = CleanUpTask("green", "red")
    room1 = CleanUpRoom("room1", [(x, y) for x in range(5) for y in range(3)], "blue")
    block1 = CleanUpBlock("block1", 1, 1, color="green")
    room2 = CleanUpRoom("room2", [(x, y) for x in range(5, 10) for y in range(3)], color="red")
    room3 = CleanUpRoom("room3", [(x, y) for x in range(0, 10) for y in range(3, 6)], color="yellow")
    rooms = [room1, room2, room3]
    blocks = [block1]
    doors = [CleanUpDoor(4, 0), CleanUpDoor(3, 2)]
    mdp = CleanUpMDP(task, rooms=rooms, doors=doors, blocks=blocks)
    mdp.visualize_interaction()

    '''
    #### testing old 2D visualisation
    from simple_rl.tasks.cleanup.cleanup_block import CleanUpBlock
    from simple_rl.tasks.cleanup.cleanup_door import CleanUpDoor
    from simple_rl.tasks.cleanup.cleanup_room import CleanUpRoom
    from simple_rl.tasks.cleanup.cleanup_task import CleanUpTask
    from simple_rl.tasks.cleanup.CleanupMDPClass import CleanUpMDP
    task = CleanUpTask("green", "red")
    room1 = CleanUpRoom("room1", [(x, y) for x in range(5) for y in range(3)], "blue")
    block1 = CleanUpBlock("block1", 1, 1, color="green")
    block2 = CleanUpBlock("block2", 2, 4, color="purple")
    block3 = CleanUpBlock("block3", 8, 1, color="orange")
    room2 = CleanUpRoom("room2", [(x, y) for x in range(5, 10) for y in range(3)], color="red")
    room3 = CleanUpRoom("room3", [(x, y) for x in range(0, 10) for y in range(3, 6)], color="yellow")
    rooms = [room1, room2, room3]
    blocks = [block1, block2, block3]
    doors = [CleanUpDoor(4, 0), CleanUpDoor(3, 2)]
    mdp = CleanUpMDP(task, rooms=rooms, doors=doors, blocks=blocks)
    #mdp.visualize_interaction()
    #mdp.visualize_value()
    '''

    return None
if __name__ == "__main__":
    A = LTLGridWorldMDP()
    ltl_visualiser(A)
