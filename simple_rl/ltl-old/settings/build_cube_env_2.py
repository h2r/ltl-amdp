import numpy as np
from collections import defaultdict

def insert_value(ii, jj, z, room_up_down, room_len, num_x, num_y):
    r_number = int(np.ceil(jj / room_len) + num_x * np.floor(ii / room_len)+ (num_x*num_y) * np.floor(z / 2))
    if r_number in room_up_down:
        return r_number
    else:
        return 'w'

def build_cube_env():
    # Large
    cube_env = {} # Define settings as a dictionary
    cube_env['len_x'] = 30  # the number of grids (x-axis)
    cube_env['len_y'] = 20  # the number of grids (y-axis)
    cube_env['len_z'] = 12  # the number of grids (z-axis)
    cube_env['num_floor'] = 6 # the number of floors
    cube_env['num_room'] = 3*2*6 # the number of rooms
    num_x = 3
    num_y = 2
    room_height = 2
    room_len = 10
    num_rooms_on_floor = num_x*num_y

    # Define a map : room number, w (wall)
    room_up_down = [num_rooms_on_floor * ii - 2 for ii in range(1, cube_env['num_floor']+1)]

    map = [] # map[z][y][x]
    for z in range(0, cube_env['len_z']):
        room_num_array = []
        for ii in range(0, cube_env['len_y']):
            room_num_array.append([int(np.ceil(jj / room_len) + num_x * np.floor(ii/ room_len)) for jj in range(1, cube_env['len_x']+1)])

        wall_array = []
        for ii in range(0, cube_env['len_y']):
            wall_array.append([insert_value(ii, jj, z, room_up_down, room_len, num_x, num_y) for jj in range(1, cube_env['len_x']+1)])

        if z in [room_height*ii-1 for ii in range(1, cube_env['num_floor']+1)]:
            map.append(wall_array)
        else:
            map.append(room_num_array + num_rooms_on_floor * np.floor(z / room_height))

    z_to_floor = [int(np.floor(z/room_height)+1.0) for z in range(0, cube_env['len_z'])]
    cube_env['map'] = map

    # --------------------- Automatically computed --------- #
    # extract (x,y,z) in each room
    room_to_locs = defaultdict()
    loc_to_room = {}
    for r in range(1,cube_env['num_room']+1):
        locs = []
        for x in range(1, cube_env['len_x']+1):
            for y in range(1,cube_env['len_y']+1):
                for z in range(1,cube_env['len_z']+1):
                    if cube_env['map'][z-1][y-1][x-1] == r:
                        locs.append((x,y,z))
                        loc_to_room[(x,y,z)] = r

        room_to_locs[r] = locs

    cube_env['room_to_locs'] = room_to_locs
    cube_env['loc_to_room'] = loc_to_room

    # extract (x,y,z) ᅟin walls
    walls = []
    for x in range(1, cube_env['len_x'] + 1):
        for y in range(1, cube_env['len_y'] + 1):
            for z in range(1, cube_env['len_z'] + 1):
                if cube_env['map'][z - 1][y - 1][x - 1] == 'w':
                    walls.append((x, y, z))

    cube_env['walls'] = walls

    # Extract room numbers and locations in each floor
    floor_to_room = defaultdict()
    floor_to_locs = defaultdict()
    room_to_floor = {}
    loc_to_floor ={}

    for f in range(1, cube_env['num_floor']+1):
        rooms = []
        locs = []
        for x in range(1, cube_env['len_x'] + 1):
            for y in range(1, cube_env['len_y'] + 1):
                for z in range(1, cube_env['len_z'] + 1):
                    room_number = cube_env['map'][z-1][y-1][x-1]
                    if  room_number not in rooms and z_to_floor[z-1] == f and room_number !='w':
                        rooms.append(room_number)
                        room_to_floor[room_number] = f
                    if z_to_floor[z - 1] == f and room_number != 'w':
                        locs.append((x, y, z))
                        loc_to_floor[(x,y,z)] = f

        floor_to_room[f] = rooms
        floor_to_locs[f] = locs

    cube_env['floor_to_rooms'] = floor_to_room
    cube_env['floor_to_locs'] = floor_to_locs
    cube_env['room_to_floor'] = room_to_floor
    cube_env['loc_to_floor']=loc_to_floor

    # Define transition table (connectivity between rooms)
    cube_env['transition_table'] = {}

    for r in range(1, cube_env['num_room']+1):
        connected_rooms = []
        for x,y,z in cube_env['room_to_locs'][r]:
            near = [(max(x-2,0), y-1, z-1), (min(x, cube_env['len_x']-1), y-1, z-1),
                    (x-1, max(y-2,0), z-1), (x-1, min(y, cube_env['len_y']-1), z-1)]
            for i,j,k in near:
                next_r = cube_env['map'][k][j][i]
                if next_r not in connected_rooms and next_r != r:
                    connected_rooms.append(next_r)

        if r in room_up_down:
            idx = room_up_down.index(r)
            if (idx-1) >= 0:
                connected_rooms.append(room_up_down[idx-1])
            if (idx+1) <= len(connected_rooms)+1:
                connected_rooms.append(room_up_down[idx+1])



        cube_env['transition_table'][r] = connected_rooms

    # Define attributes
    cube_env['attribute_color'] = {1: 'red', 6: 'blue', 12: 'blue', 18: 'blue',
                                   8: 'yellow', 10: 'purple', 15: 'green'
                                   }

    # Define Actions
    cube_env['L2ACTIONS'] = ["toFloor%d" % ii for ii in range(1, cube_env['num_floor']+1)]
    cube_env['L1ACTIONS'] = ["toRoom%d" % ii for ii in range(1, cube_env['num_room'] + 1)]
    cube_env['L0ACTIONS'] = ["north", "south", "east", "west", "up", "down"]

    # save
    #np.save('cube_env_1.npy',cube_env)

    return cube_env

if __name__ == '__main__':
    env = build_cube_env()
    print("done")