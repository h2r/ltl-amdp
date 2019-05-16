# Python imports.
from __future__ import print_function
from collections import defaultdict

try:
    import pygame
except ImportError:
    print("Warning: pygame not installed (needed for visuals).")
import random
import sys

# Other imports.
# new version for ltl
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_visualizer as mdpv
import numpy as np
import os, json

def draw_state(screen,
               cleanup_mdp,
               state,
               policy=None,
               action_char_dict={},
               show_value=False,
               agent=None,
               draw_statics=False,
               agent_shape=None):

    #print('\n\n\n\nInside draw state\n')
    # Make value dict.
    val_text_dict = defaultdict(lambda: defaultdict(float))

    # Make policy dict.
    policy_dict = defaultdict(lambda: defaultdict(str))

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0)  # Add 30 for title.

    width = cleanup_mdp.width
    height = cleanup_mdp.height

    cell_width = (scr_width - width_buffer * 2) / width
    cell_height = (scr_height - height_buffer * 2) / height


    cell_width -= 10; cell_height -= 10
    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)
    cc_font = pygame.font.SysFont("Courier", font_size * 2 + 2)

    # room_locs = [(x + 1, y + 1) for room in cleanup_mdp.rooms for (x, y) in room.points_in_room]
    door_locs = set([(door.x + 1, door.y + 1) for door in state.doors])


    def load_cube():
        a = np.load(os.getcwd() + '/AMDP/cube_env_1.npy').item()

        rooms, fin_dict = a['room_to_locs'], {}
        idx_to_colour = a['attribute_color']
        colour_to_idx = {}
        for item in idx_to_colour.items():
            colour, idx = item[1], item[0]
            if colour not in colour_to_idx.keys():
                colour_to_idx[colour] = []
            colour_to_idx[colour].append(idx)


        for colour in colour_to_idx:
            fin_dict[colour] = {0: [], 1: [], 2: []}
        for key in rooms:
            if key not in idx_to_colour:
                colour = 'None'
            else:
                colour = idx_to_colour[key]

            if colour == 'None': continue

            points_3d = rooms[key]
            floor = points_3d[0][2] - 1


            if floor == 0: floor = 2
            elif floor == 2: floor = 0
            points_2d = [(item[0]-1, item[1]-1) for item in points_3d]

            fin_dict[colour][floor] = points_2d
        return fin_dict


    def load_points_from_state_seq():
        # load states into list of tuples (x, y, floor)

        '''
        points = 's: (1,1,1)\ns: (2,1,1)\ns: (3,1,1)\ns: (4,1,1)\ns: (5,1,1)\ns: (5,2,1)\ns: (5,3,1)\ns: (5,4,1)\ns: (5,3,1)\ns: (6,3,0)\ns: (5,3,0)\ns: (4,3,0)\ns: (3,3,0)\ns: (3,4,0)\ns: (2,4,0)\ns: (2,3,0)\ns: (2,2,2)'.split('\n')

        # changed slightly
        points = 's: (1,1,1)\ns: (2,1,1)\ns: (3,1,1)\ns: (4,1,1)\ns: (5,1,1)\ns: (5,2,1)\ns: (5,3,1)\ns: (5,4,1)\ns: (5,3,1)\ns: (6,3,0)\ns: (5,3,0)\ns: (4,3,0)\ns: (3,3,0)\ns: (3,4,0)\ns: (2,4,0)\ns: (2,3,0)\ns: (2,2,0)'.split('\n')

        points = [item.split(')')[0].split('(')[-1].split(',') for item in points]
        points = [[int(val) for val in item] for item in points]
        '''


        #f = open(os.getcwd() + '/results/state_seq.txt', 'r')
        f = open('/Users/romapatel/Desktop/actions.tsv', 'r')
        points = f.readlines()

        points = [item.split(')')[0].split('(')[-1].split(',') for item in points]
        points = [[int(val) for val in item] for item in points]
        
        return points

    def draw_levels(i, j, level, cell_width, cell_height, width_buffer, height_buffer):
        x, y =  width_buffer + cell_height * i,  height_buffer + cell_width * j

        theta = math.pi/3
        #length = 80;  breadth = 40;
        length = cell_width; breadth = cell_height
        length = cell_height; breadth = cell_width
        p1 = (x, y); p2 = (x + length, y)
        p3 = (x + length - abs((breadth*math.cos(theta))), y + abs(int(breadth*math.sin(theta))))
        p4 = (x - abs(int(breadth*math.cos(theta))), y + abs(int(breadth*math.sin(theta))))

        point_list = [p1, p2, p3, p4]
        r = pygame.draw.lines(screen, (46, 49, 49), True, point_list, 1)



    # ROMA -- keep a rooma dictionary!
    import math
    def draw_3D_room(i, j, level, cell_width, cell_height, width_buffer, height_buffer, theta, curr_x, curr_y, colour, room_name):
        #print('Drawing room')

        #print(colour)
        x, y = curr_x, curr_y

        length = cell_height; breadth = cell_width
        p1 = (x, y); p2 = (x + length, y)
        p3 = (x + length - abs((breadth*math.cos(theta))), y + abs(int(breadth*math.sin(theta))))
        p4 = (x - abs(int(breadth*math.cos(theta))), y + abs(int(breadth*math.sin(theta))))

        point_list = [p1, p2, p3, p4]
        inner_points = [(p1[0] + 10, p1[1] + 10), (p2[0] - 15, p2[1] + 10), (p3[0] - 10, p3[1] -10), (p4[0] + 15, p4[1] - 10)]

        thickness = 2
        if room_name == 'room_23':
            colour = get_rgb('blue')


        pygame.draw.lines(screen, colour, True, point_list, thickness)
        fill_room(screen, colour, point_list)

        # connect elevator
        for point in point_list:
            points = [point, (point[0], point[1] - 200*level)]

            if colour == (0, 0, 255):
                pygame.draw.lines(screen, colour, True, points, 10)

        return p1, p2, p3, p4


    def fill_room(screen, colour, old_points):
        while True:
            x1, x2, x3, x4 = old_points[0][0], old_points[1][0], old_points[2][0], old_points[3][0]
            y1, y2, y3, y4 = old_points[0][1], old_points[1][1], old_points[2][1], old_points[3][1]

            if x1 >= x2 or y1 >= y3: break
            if x4 >= x3 or y2 >= y4: break

            x1 += 1; x2 -=1; x3 -=1; x4 +=1; 
            y1 += 1; y3 -=1; y2 += 1; y4 -= 1

            '''
            old_points[0][0] += 1; old_points[3][0] += 1
            old_points[1][0] -=1; old_points[2][0] -= 1

            old_points[0][1] += 1; old_points[3][1] -= 1
            old_points[1][1] +=1; old_points[2][1] -= 1
            '''

            old_points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            pygame.draw.lines(screen, colour, True, old_points, 1)


    # (x, y, level)
    colour_dict = {'red': {2: [(0, 2), (1, 2), (0, 1), (1, 1)], 1: [], 0: []}, 'green': {2: [], 1: [], 0: []}, 'yellow': {2: [], 1: [], 0: []}}


    colour_dict = load_cube()
    #print(colour_dict)
    
    def draw_floor(level):
        #print('\nDrawing floor\n')
        rooms_width, rooms_height, room_dict = 6, 4, {}

        theta = math.pi/3
        i, j = 0, 0
        curr_x, prev_x, curr_y, prev_y = 72, 72, 102, 102


        for i in range(rooms_width):
            for j in range(rooms_height):
            
                #print('(i, j) ', i, j)

                if j == 0: 
                    curr_x = prev_x = width_buffer + cell_height * i + 100
                    curr_y = prev_y = height_buffer + cell_width * j + 200*level

                else:
                    #curr_x = prev_x - abs((cell_height*math.cos(theta)))
                    curr_x, curr_y = prev_x, prev_y
                colour = (46, 49, 49)
                for key in colour_dict:
                    if (i, j) in colour_dict[key][level]:
                        colour = get_rgb(key)
                room_name = 'room_' + str(rooms_height*i+j)
                #print(room_name)
                p1, p2, p3, p4 = draw_3D_room(i, j, level, cell_width, cell_height, width_buffer, height_buffer, theta, curr_x, curr_y, colour, room_name)
                #print('Last point: ', p4)

                room_dict[room_name] = {'rooms_width': i, 'rooms_height': j, 'points': [p1, p2, p3, p4]}
                prev_x, prev_y = p4[0], p4[1]

        return room_dict

    
    def get_avg(point_list):
        x, y = 0, 0
        for item in point_list:
            x += float(item[0])/4; y += float(item[1])/4
        #agent_location = (x-25, y-50)

        return (x, y)

    def get_center(room, rooms_0, rooms_1, rooms_2):
        
        theta = math.pi/3

        # for center point of room
        i, j, level = room[0]-1, room[1]-1, room[2]
        #print('(i, j) ', i, j)

        room_name = 'room_' + str(4*i + j)
        if level == 0:
            rooms = rooms_0
        elif level == 1:
            rooms = rooms_1
        elif level == 2: 
            rooms = rooms_2

        else:
            print(room)
            print(level)
            rooms = None
        #print(level) 
        #print(rooms[room_name]); print('\n\n\n\n\n')
        curr_x, prev_x, curr_y, prev_y = 72, 72, 102, 102

        
        if j == 0: 
            curr_x = prev_x = width_buffer + cell_height * i + 100
            curr_y = prev_y = height_buffer + cell_width * j + 200*level

        else:
            #curr_x, curr_y = prev_x, prev_y
            curr_x = width_buffer + cell_height * i + 100

            curr_y = height_buffer + cell_width * j + 200*level
            curr_y = height_buffer + 200*level

        
        #curr_x =  width_buffer + cell_height * i + 100
        #curr_y = height_buffer + cell_width * j + 200*level

        colour = get_rgb('pink')
    

        x, y = curr_x, curr_y

        length = cell_height; breadth = cell_width
        p1 = (x, y); p2 = (x + length, y)
        p3 = (x + length - abs((breadth*math.cos(theta))), y + abs(int(breadth*math.sin(theta))))
        p4 = (x - abs(int(breadth*math.cos(theta))), y + abs(int(breadth*math.sin(theta))))

        point_list = [p1, p2, p3, p4]

        return rooms[room_name]['points']
        return point_list

        return (x, y)



    rooms_0 = draw_floor(0)
    rooms_1 = draw_floor(1)
    rooms_2 = draw_floor(2)


    # draw path
    path =  load_points_from_state_seq()[:-1]


    #print('\nPath: ', path)
    colour = get_rgb('pink')
    # we need a list of pairs of points which will connect the path
    prev = path[0]

    #path = path [:3]
    #print('Drawing path\n\n')
    for point_3d in path[1:]:
        #print(pair)
        current = point_3d
        prev_2d = get_center(prev, rooms_0, rooms_1, rooms_2)

        current_2d = get_center(current, rooms_0, rooms_1, rooms_2)
        '''
        print('prev: ', prev)
        print('prev 2d: ', prev_2d); 
        print('current: ', current); print('current 2d: ', <D-k>current_2d); print()
        '''

        # print rectangles over every square
        #point_list = get_center(current, rooms_0, rooms_1, rooms_2)
        #pygame.draw.lines(screen, colour, True, point_list, 3)

        # print a connecting line for path
        prev_point_list = get_center(prev, rooms_0, rooms_1, rooms_2)
        current_point_list = get_center(current, rooms_0, rooms_1, rooms_2)
        prev_avg = get_avg(prev_point_list)
        current_avg = get_avg(current_point_list)
        point_list = [prev_avg, current_avg]
        pygame.draw.lines(screen, colour, True, point_list, 3)

        prev = current
        


    # ROMA --- add agent at last action
    # draw agent

    last_location = get_center(path[-1], rooms_0, rooms_1, rooms_2)
    print(path); print(path[-1]); print(last_location)
    x, y = 0, 0
    for item in last_location:
        x += float(item[0])/4; y += float(item[1])/4
    agent_location = (x-25, y-50)

    #print('\nAgent location: ', agent_location)
    fname = os.getcwd() + '/figures/robot4.png'
    agent = pygame.transform.scale(pygame.image.load(fname), (60, 60))
    screen.blit(agent, agent_location)


    return None




    for i in range(width):
        # For each column:
        for j in range(height):

            top_left_point = width_buffer + cell_width * i, height_buffer + cell_height * j
            r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

            # if grid_mdp.is_wall(i+1, grid_mdp.height - j):
            if (i + 1, height - j) not in cleanup_mdp.legal_states:
                # Draw the walls.
                top_left_point = width_buffer + cell_width * i + 5, height_buffer + cell_height * j + 5
                pygame.draw.rect(screen, (94, 99, 99), top_left_point + (cell_width - 10, cell_height - 10), 0)

            
            if (i + 1, height - j) in door_locs:
                # Draw door
                # door_color = (66, 83, 244)
                door_color = (0, 0, 0)
                top_left_point = width_buffer + cell_width * i + 5, height_buffer + cell_height * j + 5
                pygame.draw.rect(screen, door_color, top_left_point + (cell_width - 10, cell_height - 10), 0)

            else:
                room = cleanup_mdp.check_in_room(state.rooms, i + 1 - 1, height - j - 1)  # Minus 1 for inconsistent x, y
                if room:
                    top_left_point = width_buffer + cell_width * i + 5, height_buffer + cell_height * j + 5
                    room_rgb = _get_rgb(room.color)
                    pygame.draw.rect(screen, room_rgb, top_left_point + (cell_width - 10, cell_height - 10), 0)

            block = cleanup_mdp.find_block(state.blocks, i + 1 - 1, height - j - 1)
            # print(state)
            # print(block)

    pygame.display.flip()

    return agent_shape


def old_draw_state(screen,
               cleanup_mdp,
               state,
               policy=None,
               action_char_dict={},
               show_value=False,
               agent=None,
               draw_statics=False,
               agent_shape=None):
    '''
    Args:
        screen (pygame.Surface)
        grid_mdp (MDP)
        state (State)
        show_value (bool)
        agent (Agent): Used to show value, by default uses VI.
        draw_statics (bool)
        agent_shape (pygame.rect)

    Returns:
        (pygame.Shape)
    '''

    print('Inside draw state\n\n\n\n')
    # Make value dict.
    val_text_dict = defaultdict(lambda: defaultdict(float))
    if show_value:
        if agent is not None:
            # Use agent value estimates.
            for s in agent.q_func.keys():
                val_text_dict[s.x][s.y] = agent.get_value(s)
        else:
            # Use Value Iteration to compute value.
            vi = ValueIteration(cleanup_mdp)
            vi.run_vi()
            for s in vi.get_states():
                val_text_dict[s.x][s.y] = vi.get_value(s)

    # Make policy dict.
    policy_dict = defaultdict(lambda: defaultdict(str))
    if policy:
        vi = ValueIteration(cleanup_mdp)
        vi.run_vi()
        for s in vi.get_states():
            policy_dict[s.x][s.y] = policy(s)

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0)  # Add 30 for title.

    width = cleanup_mdp.width
    height = cleanup_mdp.height

    cell_width = (scr_width - width_buffer * 2) / width
    cell_height = (scr_height - height_buffer * 2) / height
    # goal_locs = grid_mdp.get_goal_locs()
    # lava_locs = grid_mdp.get_lavacc_locs()
    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)
    cc_font = pygame.font.SysFont("Courier", font_size * 2 + 2)

    # room_locs = [(x + 1, y + 1) for room in cleanup_mdp.rooms for (x, y) in room.points_in_room]
    door_locs = set([(door.x + 1, door.y + 1) for door in state.doors])

    # Draw the static entities.
    # print(draw_statics)
    # draw_statics = True
    # if draw_statics:
        # For each row:



    for i in range(width):
        # For each column:
        for j in range(height):

            top_left_point = width_buffer + cell_width * i, height_buffer + cell_height * j
            r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

            '''
            # if policy and not grid_mdp.is_wall(i+1, height - j):
            if policy and (i + 1, height - j) in cleanup_mdp.legal_states:
                a = policy_dict[i + 1][height - j]
                if a not in action_char_dict:
                    text_a = a
                else:
                    text_a = action_char_dict[a]
                text_center_point = int(top_left_point[0] + cell_width / 2.0 - 10), int(
                    top_left_point[1] + cell_height / 3.0)
                text_rendered_a = cc_font.render(text_a, True, (46, 49, 49))
                screen.blit(text_rendered_a, text_center_point)

            # if show_value and not grid_mdp.is_wall(i+1, grid_mdp.height - j):
            if show_value and (i + 1, height - j) in cleanup_mdp.legal_states:
                # Draw the value.
                val = val_text_dict[i + 1][height - j]
                color = mdpv.val_to_color(val)
                pygame.draw.rect(screen, color, top_left_point + (cell_width, cell_height), 0)
                # text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/7.0)
                # text = str(round(val,2))
                # text_rendered = reg_font.render(text, True, (46, 49, 49))
                # screen.blit(text_rendered, text_center_point)
            '''

            # if grid_mdp.is_wall(i+1, grid_mdp.height - j):
            if (i + 1, height - j) not in cleanup_mdp.legal_states:
                # Draw the walls.
                top_left_point = width_buffer + cell_width * i + 5, height_buffer + cell_height * j + 5
                pygame.draw.rect(screen, (94, 99, 99), top_left_point + (cell_width - 10, cell_height - 10), 0)

            
            if (i + 1, height - j) in door_locs:
                # Draw door
                # door_color = (66, 83, 244)
                door_color = (0, 0, 0)
                top_left_point = width_buffer + cell_width * i + 5, height_buffer + cell_height * j + 5
                pygame.draw.rect(screen, door_color, top_left_point + (cell_width - 10, cell_height - 10), 0)

            else:
                room = cleanup_mdp.check_in_room(state.rooms, i + 1 - 1, height - j - 1)  # Minus 1 for inconsistent x, y
                if room:
                    top_left_point = width_buffer + cell_width * i + 5, height_buffer + cell_height * j + 5
                    room_rgb = _get_rgb(room.color)
                    pygame.draw.rect(screen, room_rgb, top_left_point + (cell_width - 10, cell_height - 10), 0)

            block = cleanup_mdp.find_block(state.blocks, i + 1 - 1, height - j - 1)
            # print(state)
            # print(block)


            '''
            # ROMA: to draw objects if needed
            if block:
                circle_center = int(top_left_point[0] + cell_width / 2.0), int(top_left_point[1] + cell_height / 2.0)
                block_rgb = _get_rgb(block.color)
                pygame.draw.circle(screen, block_rgb, circle_center, int(min(cell_width, cell_height) / 4.0))
            
            # Current state.
            # ROMA: to draw the agent if needed
            if not show_value and (i + 1, height - j) == (state.x + 1, state.y + 1) and agent_shape is None:
                tri_center = int(top_left_point[0] + cell_width / 2.0), int(top_left_point[1] + cell_height / 2.0)
                agent_shape = _draw_agent(tri_center, screen, base_size=min(cell_width, cell_height) / 2.5 - 8)
            '''

    '''
    if agent_shape is not None:
        # Clear the old shape.
        pygame.draw.rect(screen, (255, 255, 255), agent_shape)
        top_left_point = width_buffer + cell_width * ((state.x + 1) - 1), height_buffer + cell_height * (
                height - (state.y + 1))
        tri_center = int(top_left_point[0] + cell_width / 2.0), int(top_left_point[1] + cell_height / 2.0)

        # Draw new.
        # if not show_value or policy is not None:
        agent_shape = _draw_agent(tri_center, screen, base_size=min(cell_width, cell_height) / 2.5 - 16)
    '''
    pygame.display.flip()

    return agent_shape


def _draw_agent(center_point, screen, base_size=20):
    '''
    Args:
        center_point (tuple): (x,y)
        screen (pygame.Surface)

    Returns:
        (pygame.rect)
    '''
    tri_bot_left = center_point[0] - base_size, center_point[1] + base_size
    tri_bot_right = center_point[0] + base_size, center_point[1] + base_size
    tri_top = center_point[0], center_point[1] - base_size
    tri = [tri_bot_left, tri_top, tri_bot_right]
    tri_color = (98, 140, 190)
    return pygame.draw.polygon(screen, tri_color, tri)


def _get_rgb(color):
    '''
    :param color: A String
    :return: triple that represents the rbg color
    '''
    color = color.lower().strip()
    if color == "red":
        return 255, 0, 0
    if color == "blue":
        return 0, 0, 255
    if color == "green":
        return 0, 255, 0
    if color == "yellow":
        return 255, 255, 0
    if color == "purple":
        return 117, 50, 219
    if color == "orange":
        return 237, 138, 18
    if color == "pink":
        return 247, 2, 243

def get_rgb(color):
    '''
    :param color: A String
    :return: triple that represents the rbg color
    '''
    color = color.lower().strip()
    if color == "red":
        return 255, 0, 0
    if color == "blue":
        return 0, 0, 255
    if color == "green":
        return 0, 255, 0
    if color == "yellow":
        return 255, 255, 0
    if color == "purple":
        return 117, 50, 219
    if color == "orange":
        return 237, 138, 18
    if color == "pink":
        return 247, 2, 243
    if color == "black":
        return 46, 49, 49
    if color == "white":
        return 0, 0, 0


