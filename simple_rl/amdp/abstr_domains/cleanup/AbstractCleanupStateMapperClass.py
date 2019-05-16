# Other imports.
from simple_rl.amdp.AMDPStateMapperClass import AMDPStateMapper
from simple_rl.tasks.cleanup.cleanup_state import CleanUpState
from simple_rl.tasks.cleanup.cleanup_block import CleanUpBlock
from simple_rl.tasks.cleanup.cleanup_door import CleanUpDoor
from simple_rl.tasks.cleanup.cleanup_room import CleanUpRoom
from simple_rl.amdp.abstr_domains.cleanup.AbstractCleanupL1StateClass import *

class AbstractCleanupL1StateMapper(AMDPStateMapper):
    def __init__(self, l0_domain):
        AMDPStateMapper.__init__(self, l0_domain)
        self.l0_domain = l0_domain

    def map_state(self, state):
        '''
        Args:
            state (CleanUpState)

        Returns:
            projected_state (CleanupL1State)
        '''
        l1_robot = self._derive_l1_robot(state)
        l1_doors = self._derive_l1_doors(state)
        l1_rooms = self._derive_l1_rooms(state)
        l1_blocks = self._derive_l1_blocks(state)
        return CleanupL1State(l1_robot, l1_doors, l1_rooms, l1_blocks)

    # -----------------------------
    # ----- Helper Methods --------
    # -----------------------------

    def _derive_l1_blocks(self, state):
        '''
        Args:
            state (CleanUpState)

        Returns:
            l1_blocks (list): list of CleanupL1Block objects
        '''
        l1_blocks = []
        for block in state.blocks:  # type: CleanUpBlock
            room_color = self._position_to_room_color(state.rooms, (block.x, block.y))
            block_color = block.color
            current_door = self._get_block_door(state, block)
            l1_blocks.append(CleanupL1Block(room_color, current_door, block_color))
        return l1_blocks

    def _derive_l1_rooms(self, state):
        '''
        Args:
            state (CleanUpState)

        Returns:
            l1_rooms (list)
        '''
        def _doors_in_room(s, room_points, doors):
            door_names = [self._get_connecting_rooms_from_door(s, dr) for dr in doors if (dr.x, dr.y) in room_points]
            return [connecting_rooms[0] + '_' + connecting_rooms[1] for connecting_rooms in door_names]
        return [CleanupL1Room(room.color, _doors_in_room(state, room.points_in_room, state.doors)) for room in state.rooms]

    def _derive_l1_doors(self, state):
        '''
        Args:
            state (CleanUpState)

        Returns:
            l1_doors (list): list of CleanupL1Door Objects
        '''
        def determine_room(rooms, xpos, ypos):
            for room in rooms: # type: CleanUpRoom
                if (xpos, ypos) in room.points_in_room:
                    return room.color
            raise ValueError('Unable to find the room corresponding to door at {}'.format((xpos, ypos)))

        l1_doors = []
        for door in state.doors: # type: CleanUpDoor
            connecting_rooms = self._get_connecting_rooms_from_door(state, door)
            if connecting_rooms:
                l1_doors.append(CleanupL1Door(connecting_rooms, determine_room(state.rooms, door.x, door.y)))
        return l1_doors

    def _derive_l1_robot(self, state):
        '''
        Args:
            state (CleanUpState)

        Returns:
            robot (CleanupL1Robot)
        '''
        robot_room = self._position_to_room_color(state.rooms, (state.x, state.y))
        adjacent_block_color = None
        robot_current_door = ''
        for block in state.blocks:
            if AbstractCleanupL1StateMapper._is_block_adjacent_to_robot(state, block):
                adjacent_block_color = block.color
        for door in state.doors: # type: CleanUpDoor
            connecting_rooms = self._get_connecting_rooms_from_door(state, door)
            if connecting_rooms:
                room1, room2 = connecting_rooms[0], connecting_rooms[1]
                if state.x == door.x and state.y == door.y:
                    robot_current_door = room1 + '_' + room2
        return CleanupL1Robot(robot_room, robot_current_door, adjacent_block_color)

    @staticmethod
    def _position_to_room_color(rooms, position):
        '''
        Args:
            rooms (list) of CleanupRoom objects
            position (tuple)

        Returns:
            room_color (str)
        '''
        for room in rooms: # type: CleanUpRoom
            if position in room.points_in_room:
                return room.color
        return None

    @staticmethod
    def _is_block_adjacent_to_robot(state, block):
        '''
        Args:
            state (CleanUpState)
            block (CleanUpBlock)

        Returns:
            is_adjacent (bool): true if the agent is horizontally or vertically adjacent to the block
        '''
        manhattan_distance = abs(state.x - block.x) + abs(state.y - block.y)
        return manhattan_distance <= 1

    def _get_connecting_rooms_from_door(self, state, door):
        '''
        Args:
            state (CleanUpState)
            door (CleanUpDoor)

        Returns:
            connecting_rooms (list): [source_room_color, destination_room_color]
        '''
        connecting_rooms = []
        left_room = self._position_to_room_color(state.rooms, (door.x - 1, door.y))
        right_room = self._position_to_room_color(state.rooms, (door.x + 1, door.y))
        above_room = self._position_to_room_color(state.rooms, (door.x, door.y + 1))
        below_room = self._position_to_room_color(state.rooms, (door.x, door.y - 1))
        if left_room and right_room and left_room != right_room:
            connecting_rooms = [left_room, right_room]
        elif above_room and below_room and above_room != below_room:
            connecting_rooms = [above_room, below_room]
        return connecting_rooms

    def _get_block_door(self, state, block):
        '''
        Args:
            state (CleanUpState)
            block (CleanUpBlock)

        Returns:
            door_str: string representing the door in which the block is currently present
        '''
        current_door = ''
        for door in state.doors:  # type: CleanUpDoor
            if block.x == door.x and block.y == door.y:
                connecting_rooms = self._get_connecting_rooms_from_door(state, door)
                current_door = connecting_rooms[0] + '_' + connecting_rooms[1]
        return current_door

