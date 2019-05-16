''' FourRoomMDPClass.py: Contains the FourRoom class. '''

# Python imports.
import math
from collections import defaultdict
import numpy as np

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class FourRoomMDP(GridWorldMDP):
    ''' Class for a FourRoom '''

    def __init__(self, width=9, height=9, init_loc=(1,1), goal_locs=[(9,9)], gamma=0.99, slip_prob=0.00, name="four_room", is_goal_terminal=True, rand_init=False, step_cost=0.0):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
        '''
        GridWorldMDP.__init__(self, width, height, init_loc, goal_locs=goal_locs, walls=self._compute_walls(width, height), gamma=gamma, slip_prob=slip_prob, name=name, is_goal_terminal=is_goal_terminal, rand_init=rand_init, step_cost=step_cost)

        self.room_to_locs = defaultdict()
        for i in range(1, 5):
            self.room_to_locs[i] = self._locs_in_room(i)

    def is_loc_in_room(self, loc, room_number):
        return loc in self.room_to_locs[room_number]

    def get_room_numbers(self, loc):
        room_numbers = []
        for i in range(1, 5):
            if loc in self.room_to_locs[i]:
                room_numbers.append(i)
        return room_numbers

    def _compute_walls(self, width, height):
        '''
        Args:
            width (int)
            height (int)

        Returns:
            (list): Contains (x,y) pairs that define wall locations.
        '''
        walls = []

        half_width = math.ceil(width / 2.0)
        half_height = math.ceil(height / 2.0)

        # Wall from left to middle.
        for i in range(1, width + 1):
            if i == (width + 1) / 3 or i == math.ceil(2 * (width + 1) / 3.0):
                continue
            walls.append((i, half_height))

        # Wall from bottom to top.
        for j in range(1, height + 1):
            if j == (height + 1) / 3 or j == math.ceil(2 * (height + 1) / 3.0):
                continue
            walls.append((half_width, j))

        return walls

    def _locs_in_room(self, room_number):
        locs = []
        half_width = int(math.ceil(self.width / 2.0))
        half_height = int(math.ceil(self.height / 2.0))

        start_width, end_width = 1, half_width
        start_height, end_height = 1, half_height - 1

        if room_number == 2:
            start_width = half_width + 1
            end_width = self.width
            end_height += 1
        elif room_number == 3:
            start_height = half_height
            end_height = self.height
            end_width -= 1
        elif room_number == 4:
            start_width = half_width
            end_width = self.width
            start_height = half_height + 1
            end_height = self.height

        for x in range(start_width, end_width+1):
            for y in range(start_height, end_height+1):
                if (x, y) not in self.walls:
                    locs.append((x, y))
        return locs

if __name__ == '__main__':
    mdp = FourRoomMDP(width=5, height=5)
