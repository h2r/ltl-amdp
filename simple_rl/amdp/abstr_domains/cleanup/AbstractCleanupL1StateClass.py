# Other imports.
from simple_rl.mdp.StateClass import State

class CleanupL1State(State):
    def __init__(self, robot, doors, rooms, blocks):
        '''
        Args:
            robot (CleanupL1Robot)
            doors (list): list of all the CleanupL1Door objects
            rooms (list): list of all the CleanupL1Room objects
            blocks (list): list of all the CleanupL1Block objects
        '''
        self.robot = robot
        self.doors = doors
        self.rooms = rooms
        self.blocks = blocks

        State.__init__(self, data=[robot, doors, rooms, blocks])

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.robot == other.robot and self.doors == other.doors and \
                self.rooms == other.rooms and self.blocks == other.blocks

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return str(self.robot) + '\t' + str(self.doors) + '\t' + str(self.rooms) + '\t' + str(self.blocks) + '\n'

    def __repr__(self):
        return self.__str__()

    def get_l1_block_for_color(self, block_color):
        '''
        Args:
            block_color (str)

        Returns:
            block (CleanupL1Block)
        '''
        for block in self.blocks:
            if block.block_color == block_color:
                return block
        return None

    def get_l1_door_for_color(self, door_name):
        '''
        Args:
            door_name (str): '<room1_room2>

        Returns:
            door (CleanupL1Door)
        '''
        for door in self.doors: # type: CleanupL1Door
            if door.connected_rooms[0] in door_name and door.connected_rooms[1] in door_name:
                return door
        return None

class CleanupL1Robot(object):
    def __init__(self, current_room, current_door, adjacent_block=None):
        '''
        Args:
            current_room (str): color of the agent's current room
            current_door (str): `src_dest` str representing the source and destination rooms of a door
            adjacent_block (str): color of the block next to the agent
        '''
        self.current_room = current_room
        self.current_door = current_door
        self.adjacent_block = adjacent_block

    def __str__(self):
        block = self.adjacent_block if self.adjacent_block else 'NoBlock'
        door = self.current_door if self.current_door else 'NotAtDoor'
        return 'Robot::room:' + self.current_room + '  adjacent_block:' + block + ' current_door:' + door

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.current_room == other.current_room and self.current_door == other.current_door and\
               self.adjacent_block == other.adjacent_block

    def __ne__(self, other):
        return not self == other

class CleanupL1Door(object):
    def __init__(self, connected_rooms, current_room):
        '''
        Args:
            connected_rooms (list): list of strings representing the colors of the 2 rooms connected
            by the current door
            current_room (str): color of the room in which the current door is placed
        '''
        self.connected_rooms = connected_rooms
        self.current_room = current_room

    def __str__(self):
        return str(self.connected_rooms[0]) + '_' + str(self.connected_rooms[1])

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.connected_rooms == other.connected_rooms and self.current_room == other.current_room

    def __ne__(self, other):
        return not self == other

class CleanupL1Room(object):
    def __init__(self, room_color, door_names):
        '''
        Args:
            room_color (str): color of the current room
            door_names (list): list of strings <door1_color__door2_color>
        '''
        self.room_color = room_color
        self.door_names = door_names

    def __str__(self):
        return self.room_color

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.room_color == other.room_color and self.door_names == other.door_names

    def __ne__(self, other):
        return not self == other

class CleanupL1Block(object):
    def __init__(self, current_room, current_door, block_color):
        '''
        Args:
            current_room (str): color of the room in which the current block is placed
            current_door (str): src_dest str representing the source and destination rooms of a door
            block_color (str): color of the current block
        '''
        self.current_room = current_room
        self.current_door = current_door
        self.block_color = block_color

    def __str__(self):
        door = self.current_door if self.current_door else 'NotAtDoor'
        return 'Block::color:' + self.block_color + ' in_room:' + str(self.current_room) + ' in_door:' + str(door)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.current_room == other.current_room and self.current_door == other.current_door and\
               self.block_color == other.block_color

    def __ne__(self, other):
        return not self == other
