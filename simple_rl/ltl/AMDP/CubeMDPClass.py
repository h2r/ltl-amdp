''' GridWorldMDPClass.py: Contains the GridWorldMDP class. '''

# Python imports.
from __future__ import print_function
import random
import sys
import os
import numpy as np

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.ltl.AMDP.CubeStateClass import CubeState

# Fix input to cooperate with python 2 and 3.
try:
   input = raw_input
except NameError:
   pass

class CubeMDP(MDP):
    ''' Class for a Cube World MDP '''

    # Static constants.
    ACTIONS = ["north", "south", "west","east", "up", "down"] #""up", "down", "left", "right"]

    def __init__(self,
                len_x = 5, len_y = 5, len_z = 5,#width=5,
                #height=3,
                init_loc=(1,1,1),
                rand_init=False,
                goal_locs=[(5,3,3)],
                lava_locs=[()],
                walls=[],
                is_goal_terminal=True,
                gamma=0.99,
                init_state=None,
                slip_prob=0.0,
                step_cost=0.0,
                lava_cost=0.01,
                name="cubeworld"):
        '''
        Args:
            len_x, len_y, len_z (int): the size of state space
            init_loc (tuple: (int, int, int))
            goal_locs (list of tuples: [(int, int)...])
            lava_locs (list of tuples: [(int, int)...]): These locations return -1 reward.
        '''

        # Setup init location.
        self.rand_init = rand_init
        if rand_init:
            init_loc = random.randint(1, len_x), random.randint(1, len_y), random.randint(1, len_z)
            while init_loc in walls:
                init_loc = random.randint(1, len_x), random.randint(1, len_y), random.randint(1, len_z)
        self.init_loc = init_loc
        init_state = CubeState(init_loc[0], init_loc[1],init_loc[2]) if init_state is None or rand_init else init_state

        MDP.__init__(self, CubeMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)

        if type(goal_locs) is not list:
            raise ValueError("(simple_rl) GridWorld Error: argument @goal_locs needs to be a list of locations. For example: [(3,3), (4,3)].")
        self.step_cost = step_cost
        self.lava_cost = lava_cost
        self.walls = walls
        self.len_x = len_x
        self.len_y = len_y
        self.len_z = len_z
        self.goal_locs = goal_locs
        self.cur_state = CubeState(init_loc[0], init_loc[1], init_loc[2])
        self.is_goal_terminal = is_goal_terminal
        self.slip_prob = slip_prob
        self.name = name
        self.lava_locs = lava_locs

    def set_slip_prob(self, slip_prob):
        self.slip_prob = slip_prob

    def get_slip_prob(self):
        return self.slip_prob

    def is_goal_state(self, state):
        return (state.x, state.y) in self.goal_locs

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''
        if self._is_goal_state_action(state, action):
            return 1.0 - self.step_cost
        elif (int(state.x), int(state.y)) in self.lava_locs:
            return -self.lava_cost
        else:
            return 0 - self.step_cost

    def _is_goal_state_action(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (bool): True iff the state-action pair send the agent to the goal state.
        '''
        if (state.x, state.y, state.z) in self.goal_locs and self.is_goal_terminal:
            # Already at terminal.
            return False

        if action == "west" and (state.x - 1, state.y, state.z) in self.goal_locs:
            return True
        elif action == "east" and (state.x + 1, state.y, state.z) in self.goal_locs:
            return True
        elif action == "south" and (state.x, state.y - 1, state.z) in self.goal_locs:
            return True
        elif action == "north" and (state.x, state.y + 1, state.z) in self.goal_locs:
            return True
        elif action == "up" and (state.x, state.y, state.z+1) in self.goal_locs:
            return True
        elif action == "down" and (state.x, state.y, state.z-1) in self.goal_locs:
            return True
        else:
            return False

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        if state.is_terminal():
            return state

        r = random.random()
        if self.slip_prob > r:
            # Flip dir.
            if action == "north":
                action = random.choice(["west", "east"])
            elif action == "south":
                action = random.choice(["west", "east"])
            elif action == "west":
                action = random.choice(["north", "south"])
            elif action == "east":
                action = random.choice(["north", "south"])
            elif action == "up":
                action = random.choice(["north", "south", "east", "west"])
            elif action == "down":
                action = random.choice(["north", "south", "east", "west"])

        if action == "north" and state.y < self.len_y and not self.is_wall(state.x, state.y + 1, state.z):
            next_state = CubeState(state.x, state.y + 1, state.z)
        elif action == "south" and state.y > 1 and not self.is_wall(state.x, state.y - 1, state.z):
            next_state = CubeState(state.x, state.y - 1, state.z)
        elif action == "east" and state.x < self.len_x and not self.is_wall(state.x + 1, state.y, state.z):
            next_state = CubeState(state.x + 1, state.y, state.z)
        elif action == "west" and state.x > 1 and not self.is_wall(state.x - 1, state.y, state.z):
            next_state = CubeState(state.x - 1, state.y, state.z)
        elif action == "up" and state.z < self.len_z and not self.is_wall(state.x, state.y, state.z + 1):
            next_state = CubeState(state.x, state.y, state.z + 1)
        elif action == "down" and state.z > 1 and not self.is_wall(state.x, state.y, state.z - 1):
            next_state = CubeState(state.x, state.y, state.z - 1)

        else:
            next_state = CubeState(state.x, state.y, state.z)

        #if (next_state.x, next_state.y, next_state.z) in self.goal_locs and self.is_goal_terminal:
        #    next_state.set_terminal(True)

        return next_state

    def is_wall(self, x, y, z):
        '''
        Args:
            x (int)
            y (int)
            z (int)

        Returns:
            (bool): True iff (x,y) is a wall location.
        '''

        return (x, y, z) in self.walls

    def __str__(self):
        return self.name + "_x-" + str(self.len_x) + "_y-" + str(self.len_y) + "_z-" + str(self.len_z)

    def __repr__(self):
        return self.__str__()

    def get_goal_locs(self):
        return self.goal_locs

    def get_lava_locs(self):
        return self.lava_locs

    def visualize_policy(self, policy):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state

        action_char_dict = {
            "north":"^",       #u"\u2191",
            "south":"v",     #u"\u2193",
            "west":"<",     #u"\u2190",
            "right":">",    #u"\u2192"
            "up":"+",
            "down":"-"

        }

        mdpv.visualize_policy(self, policy, _draw_state, action_char_dict)
        input("Press anything to quit")

# TODO: visualize functions
    def visualize_agent(self, agent):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_agent(self, agent, _draw_state)
        input("Press anything to quit")

    def visualize_value(self):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_value(self, _draw_state)
        input("Press anything to quit")

    def visualize_learning(self, agent, delay=0.0):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_learning(self, agent, _draw_state, delay=delay)
        input("Press anything to quit")

    def visualize_interaction(self):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_interaction(self, _draw_state)
        input("Press anything to quit")

def _error_check(state, action):
    '''
    Args:
        state (State)
        action (str)

    Summary:
        Checks to make sure the received state and action are of the right type.
    '''

    if action not in CubeMDP.ACTIONS:
        raise ValueError("(simple_rl) CubeWorldError: the action provided (" + str(action) + ") was invalid in state: " + str(state) + ".")

    if not isinstance(state, CubeState):
        raise ValueError("(simple_rl) CubeWorldError: the given state (" + str(state) + ") was not of the correct class.")

def make_grid_world_from_file(file_name, randomize=False, num_goals=1, name=None, goal_num=None, slip_prob=0.0):
    # TODO: modify
    '''
    Args:
        file_name (str)
        randomize (bool): If true, chooses a random agent location and goal location.
        num_goals (int)
        name (str)

    Returns:
        (GridWorldMDP)

    Summary:
        Builds a GridWorldMDP from a file:
            'w' --> wall
            'a' --> agent
            'g' --> goal
            '-' --> empty
    '''

    if name is None:
        name = file_name.split(".")[0]

    # grid_path = os.path.dirname(os.path.realpath(__file__))
    wall_file = open(os.path.join(os.getcwd(), file_name))
    wall_lines = wall_file.readlines()

    # Get walls, agent, goal loc.
    num_rows = len(wall_lines)
    num_cols = len(wall_lines[0].strip())
    empty_cells = []
    agent_x, agent_y = 1, 1
    walls = []
    goal_locs = []
    lava_locs = []

    for i, line in enumerate(wall_lines):
        line = line.strip()
        for j, ch in enumerate(line):
            if ch == "w":
                walls.append((j + 1, num_rows - i))
            elif ch == "g":
                goal_locs.append((j + 1, num_rows - i))
            elif ch == "l":
                lava_locs.append((j + 1, num_rows - i))
            elif ch == "a":
                agent_x, agent_y = j + 1, num_rows - i
            elif ch == "-":
                empty_cells.append((j + 1, num_rows - i))

    if goal_num is not None:
        goal_locs = [goal_locs[goal_num % len(goal_locs)]]

    if randomize:
        agent_x, agent_y = random.choice(empty_cells)
        if len(goal_locs) == 0:
            # Sample @num_goals random goal locations.
            goal_locs = random.sample(empty_cells, num_goals)
        else:
            goal_locs = random.sample(goal_locs, num_goals)

    if len(goal_locs) == 0:
        goal_locs = [(num_cols, num_rows)]

    return CubeMDP(len_x=num_cols, len_y=num_rows, len_z = 3, init_loc=(agent_x, agent_y, 1), goal_locs=goal_locs, lava_locs=lava_locs, walls=walls, name=name, slip_prob=slip_prob)

    def reset(self):
        if self.rand_init:
            init_loc = random.randint(1, self.len_x), random.randint(1, self.len_y), random.randint(1, self.len_z)
            self.cur_state = CubeState(init_loc[0], init_loc[1], init_log[2])
        else:
            self.cur_state = copy.deepcopy(self.init_state)

def main():
    grid_world = CubeMDP(5, 10,3, (1, 1,1), (6, 7,2))

    grid_world.visualize()

if __name__ == "__main__":
    main()
