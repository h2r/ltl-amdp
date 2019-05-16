# Python imports
from __future__ import print_function
from collections import defaultdict
import copy

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.planning import ValueIteration
from simple_rl.amdp.AMDPTaskNodesClass import NonPrimitiveAbstractTask, RootTaskNode
from simple_rl.amdp.abstr_domains.cleanup.AbstractCleanupL1StateClass import *
from simple_rl.amdp.abstr_domains.cleanup.AbstractCleanupStateMapperClass import AbstractCleanupL1StateMapper
from simple_rl.tasks.cleanup.cleanup_state import CleanUpState

class CleanupL1GroundedAction(NonPrimitiveAbstractTask):
    def __init__(self, l1_action_string, subtasks, l0_domain):
        '''
        Args:
            l1_action_string (str)
            subtasks (list)
            l0_domain (CleanUpMDP)
        '''
        self.action = l1_action_string
        self.l0_domain = l0_domain
        self.lifted_action = self.grounded_to_lifted_action(l1_action_string)

        tf, rf = self._terminal_function, self._reward_function
        NonPrimitiveAbstractTask.__init__(self, l1_action_string, subtasks, tf, rf)

    def _terminal_function(self, state):
        '''
        Args:
            state (CleanUpState)

        Returns:
            is_terminal (bool)
        '''

        assert type(state) == CleanUpState, 'Actual type of state is {}'.format(type(state))

        def _robot_door_terminal_func(s, door_color):
            return s.robot.current_door == door_color
        def _robot_room_terminal_func(s, room_color):
            return s.robot.current_room == room_color and s.robot.current_door == ''
        def _robot_to_block_terminal_func(s, block_color):
            return s.robot.adjacent_block == block_color
        def _block_to_door_terminal_func(s, block_color, door_color):
            for block in s.blocks:
                if block.block_color == block_color and block.current_door == door_color:
                    return True
            return False
        def _block_to_room_terminal_func(s, block_color, room_color):
            for block in s.blocks:
                if block.block_color == block_color and block.current_room == room_color and block.current_door == '':
                    return True
            return False

        state_mapper = AbstractCleanupL1StateMapper(self.l0_domain)
        projected_state = state_mapper.map_state(state)
        action_parameter = self.grounded_to_action_parameter(self.action)

        if self.lifted_action == 'toDoor':
            return _robot_door_terminal_func(projected_state, action_parameter)
        if self.lifted_action == 'toRoom':
            return _robot_room_terminal_func(projected_state, action_parameter)
        if self.lifted_action == 'toObject':
            return _robot_to_block_terminal_func(projected_state, action_parameter)
        if self.lifted_action == 'objectToDoor':
            return _block_to_door_terminal_func(projected_state, projected_state.robot.adjacent_block, action_parameter)
        if self.lifted_action == 'objectToRoom':
            return _block_to_room_terminal_func(projected_state, projected_state.robot.adjacent_block, action_parameter)

        raise ValueError('Lifted action {} not supported yet'.format(self.lifted_action))

    def _reward_function(self, state, action):
        assert type(state) == CleanUpState, 'Actual type of state is {}'.format(type(state))
        next_state = self.l0_domain.transition_func(state, action)
        return 1. if self._terminal_function(next_state) else 0.

    # -------------------------------
    # L1 Action Helper Functions
    # -------------------------------

    @staticmethod
    def grounded_to_lifted_action(grounded_action_str):
        return grounded_action_str.split('(')[0]

    @staticmethod
    def grounded_to_action_parameter(grounded_action_str):
        return grounded_action_str.split('(')[1].split(')')[0]

    @staticmethod
    def door_name_to_room_colors(door_name):
        return door_name.split('_')

    @staticmethod
    def get_other_room_color(state, door_name):
        connected_rooms = CleanupL1GroundedAction.door_name_to_room_colors(door_name)
        if state.robot.current_room == connected_rooms[0]:
            return connected_rooms[1]
        if state.robot.current_room == connected_rooms[1]:
            return connected_rooms[0]
        return ''

class CleanupRootGroundedAction(RootTaskNode):
    def __init__(self, action_str, subtasks, l1_domain, terminal_func, reward_func):
        self.action = action_str

        RootTaskNode.__init__(self, action_str, subtasks, l1_domain, terminal_func, reward_func)

class CleanupL1MDP(MDP):
    LIFTED_ACTIONS = ['toDoor', 'toRoom', 'toObject', 'objectToDoor', 'objectToRoom']

    # -------------------------------
    # Level 1 MDP description
    # -------------------------------

    def __init__(self, l0_domain):
        '''
        Args:
            l0_domain (CleanUpMDP)
        '''
        self.l0_domain = l0_domain

        state_mapper = AbstractCleanupL1StateMapper(l0_domain)
        l1_init_state = state_mapper.map_state(l0_domain.init_state)
        grounded_actions = CleanupL1MDP.ground_actions(l1_init_state)
        self.terminal_func = self._is_goal_state

        MDP.__init__(self, grounded_actions, self._transition_function, self._reward_function, l1_init_state)

    def _is_goal_state(self, state):
        for block in state.blocks: # type: CleanupL1Block
            if block.block_color == self.l0_domain.task.block_color:
                return block.current_room == self.l0_domain.task.goal_room_color and \
                       state.robot.current_room == self.l0_domain.task.goal_room_color
        raise ValueError('Did not find an L1 Block object with color {}'.format(self.l0_domain.task.block_color))

    def _reward_function(self, state, action):
        '''
        Args:
            state (CleanupL1State)
            action (str)

        Returns:
            reward (float)
        '''
        next_state = self._transition_function(state, action)
        return 1. if self._is_goal_state(next_state) else 0.

    def _transition_function(self, state, action):
        '''
        Args:
            state (CleanupL1State)
            action (str): grounded action

        Returns:
            next_state (CleanupL1State)
        '''
        next_state = copy.deepcopy(state)
        lifted_action = CleanupL1GroundedAction.grounded_to_lifted_action(action)

        if lifted_action == 'toDoor':
            target_door_name = CleanupL1GroundedAction.grounded_to_action_parameter(action)
            next_state = self._move_agent_to_door(state, target_door_name)

        if lifted_action == 'toRoom':
            destination_room = CleanupL1GroundedAction.grounded_to_action_parameter(action)
            next_state = self._move_agent_to_room(state, destination_room)

        if lifted_action == 'toObject':
            block_color = CleanupL1GroundedAction.grounded_to_action_parameter(action)
            next_state = self._move_agent_to_block(state, block_color)

        if lifted_action == 'objectToDoor':
            target_door_name = CleanupL1GroundedAction.grounded_to_action_parameter(action)
            # next_state = self._move_agent_to_door(state, target_door_name)
            next_state = self._move_block_to_door(next_state, target_door_name)

        if lifted_action == 'objectToRoom':
            destination_room = CleanupL1GroundedAction.grounded_to_action_parameter(action)
            next_state = self._move_agent_to_room(state, destination_room)
            next_state = self._move_block_to_room(next_state, destination_room)

        next_state.set_terminal(self._is_goal_state(next_state))

        return next_state

    @classmethod
    def ground_actions(cls, l1_state):
        '''
        Given a list of lifted/parameterized actions and the L0 cleanup domain,
        generate a list of grounded actions based on the attributes of the objects
        instantiated in the L0 domain.
        Args:
            l1_state (CleanupL1State): underlying ground level MDP

        Returns:
            actions (list): grounded actions
        '''
        grounded_actions = []

        for door in l1_state.doors:  # type: CleanupL1Door
            grounded_actions.append(cls.LIFTED_ACTIONS[0] + '(' + str(door) + ')')
            grounded_actions.append(cls.LIFTED_ACTIONS[3] + '(' + str(door) + ')')

        for room in l1_state.rooms:  # type: CleanupL1Room
            grounded_actions.append(cls.LIFTED_ACTIONS[1] + '(' + str(room) + ')')
            grounded_actions.append(cls.LIFTED_ACTIONS[4] + '(' + str(room) + ')')

        for block in l1_state.blocks:  # type: CleanupL1Block
            grounded_actions.append(cls.LIFTED_ACTIONS[2] + '(' + str(block.block_color) + ')')

        return grounded_actions

    # -----------------------------------
    # Agent Navigation Helper functions
    # -----------------------------------

    @staticmethod
    def _move_agent_to_door(state, door_name):
        '''
        If the specified door connects the agent's current room, then it may transition to the door.
        Args:
            state (CleanupL1State)
            door_name (str)

        Returns:
            next_state (CleanupL1State)
        '''
        next_state = copy.deepcopy(state)
        destination_door = state.get_l1_door_for_color(door_name)
        if destination_door and state.robot.current_room in door_name:

            # If there is already a block at the door, then move it to the other room
            block = state.get_l1_block_for_color(state.robot.adjacent_block)
            if block:
                if block.current_door == door_name:
                    other_room = CleanupL1GroundedAction.get_other_room_color(state, door_name)
                    next_state = CleanupL1MDP._move_block_to_room(state, other_room)

            next_state.robot.current_door = door_name
            next_state.robot.current_room = destination_door.current_room
        return next_state

    @staticmethod
    def _move_agent_to_room(state, destination_room_color):
        '''
        Move the agent to the specified room if it is at a door connecting it to the said room.
        Args:
            state (CleanupL1State)
            destination_room_color (str)

        Returns:
            next_state (CleanupL1State)
        '''
        next_state = copy.deepcopy(state)
        if destination_room_color in state.robot.current_door:
            next_state.robot.current_room = destination_room_color
            next_state.robot.current_door = ''
        return next_state

    @staticmethod
    def _move_agent_to_block(state, block_color):
        '''
        Move the agent to the specified block if they are both in the same room.
        Args:
            state (CleanupL1State)
            block_color (str)

        Returns:
            next_state (CleanupL1State)
        '''
        next_state = copy.deepcopy(state)
        target_block = state.get_l1_block_for_color(block_color)
        if target_block:
            if target_block.current_room == state.robot.current_room:
                next_state.robot.adjacent_block = target_block.block_color
                next_state.robot.current_door = ''
        return next_state

    # -----------------------------------
    # Block Navigation Helper functions
    # -----------------------------------

    @staticmethod
    def _move_block_to_door(state, door_name):
        '''
        Move the agent's adjacent block to the specified door if they are in a room connected by said door.
        Args:
            state (CleanupL1State)
            door_name (str)

        Returns:
            next_state (CleanupL1State)
        '''
        next_state = copy.deepcopy(state)
        block = next_state.get_l1_block_for_color(next_state.robot.adjacent_block)
        destination_door = next_state.get_l1_door_for_color(door_name)
        if block and destination_door:
            if state.robot.current_room in door_name and block.current_room in door_name:
                next_state.robot.current_room = block.current_room
                next_state.robot.current_door = ''
                block.current_door = door_name
                block.current_room = destination_door.current_room
        return next_state

    @staticmethod
    def _move_block_to_room(state, destination_room_color):
        '''
        Move the block to the specified room if the block is at a door connecting said room.
        Args:
            state (CleanupL1State)
            destination_room_color (str)

        Returns:
            next_state (CleanupL1State)
        '''
        next_state = copy.deepcopy(state)
        block = next_state.get_l1_block_for_color(next_state.robot.adjacent_block)
        if block:
            if destination_room_color in block.current_door:
                block.current_room = destination_room_color
                block.current_door = ''
        return next_state

# -----------------------------------
# Debug functions
# -----------------------------------

def debug_l1_domain():
    from simple_rl.tasks.cleanup.cleanup_block import CleanUpBlock
    from simple_rl.tasks.cleanup.cleanup_door import CleanUpDoor
    from simple_rl.tasks.cleanup.cleanup_room import CleanUpRoom
    from simple_rl.tasks.cleanup.cleanup_task import CleanUpTask
    from simple_rl.tasks.cleanup.CleanupMDPClass import CleanUpMDP

    def get_l1_policy(domain):
        vi = ValueIteration(domain, sample_rate=1)
        vi.run_vi()

        policy = defaultdict()
        action_seq, state_seq = vi.plan(domain.init_state)

        print('Plan for {}:'.format(domain))
        for i in range(len(action_seq)):
            print("\tpi[{}] -> {}\n".format(state_seq[i], action_seq[i]))
            policy[state_seq[i]] = action_seq[i]

        return policy

    task = CleanUpTask("purple", "red")
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

    amdp = CleanupL1MDP(mdp)

    get_l1_policy(amdp)
