# Python imports.
from __future__ import print_function

# Generic AMDP imports.
from simple_rl.amdp.AMDPSolverClass import AMDPAgent
from simple_rl.amdp.AMDPTaskNodesClass import PrimitiveAbstractTask

# Abstract grid world imports.
from simple_rl.ltl.AMDP.RoomCubeMDPClass import RoomCubeMDP
from simple_rl.ltl.AMDP.AbstractCubeMDPClass import *
from simple_rl.ltl.AMDP.AbstractCubePolicyGeneratorClass import *
from simple_rl.ltl.AMDP.AbstractCubeStateMapperClass import *

from simple_rl.ltl.settings.build_cube_env_1 import build_cube_env

if __name__ == '__main__':
    cube_env = build_cube_env()
    start_floor = 1
    goal_floor = 3
    start_room, goal_room = 1, 15
    init_locs = cube_env['room_to_locs'][start_room][0]
    goal_locs = cube_env['room_to_locs'][goal_room]
    l0Domain = RoomCubeMDP(init_loc=init_locs, goal_locs=goal_locs, env_file=[cube_env])
    l1Domain = CubeL1MDP(start_room, goal_room, env_file=[cube_env])
    l2Domain = CubeL2MDP(start_floor, goal_floor, env_file=[cube_env])

    policy_generators = []
    l0_policy_generator = CubeL0PolicyGenerator(l0Domain, env_file = [cube_env])
    l1_policy_generator = CubeL1PolicyGenerator(l0Domain, AbstractCubeL1StateMapper(l0Domain), env_file = [cube_env])
    l2_policy_generator = CubeL2PolicyGenerator(l1Domain, AbstractCubeL2StateMapper(l1Domain), env_file=[cube_env])
    policy_generators.append(l0_policy_generator)
    policy_generators.append(l1_policy_generator)
#    policy_generators.append(l2_policy_generator)

    # 3 levels
    #l1Subtasks = [PrimitiveAbstractTask(action) for action in l0Domain.ACTIONS]
    #l2Subtasks = [CubeL1GroundedAction(a, l1Subtasks, l0Domain) for a in l1Domain.ACTIONS]
    #a2rt = [CubeL2GroundedAction(a, l2Subtasks, l1Domain) for a in l2Domain.ACTIONS]

    #l2Root = CubeRootL2GroundedAction(l2Domain.action_for_floor_number(goal_floor), a2rt, l2Domain,
    #                                  l2Domain.terminal_func, l2Domain.reward_func)

    # 2 levels
    l1Subtasks = [PrimitiveAbstractTask(action) for action in l0Domain.ACTIONS]
    a2rt = [CubeL1GroundedAction(a, l1Subtasks, l0Domain) for a in l1Domain.ACTIONS]
    l1Root = CubeRootL1GroundedAction(l1Domain.action_for_room_number(goal_room), a2rt, l1Domain,
                                        l1Domain.terminal_func, l1Domain.reward_func)

    agent = AMDPAgent(l1Root, policy_generators, l0Domain)
    agent.solve()
