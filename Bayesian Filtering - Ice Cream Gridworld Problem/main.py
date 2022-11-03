import numpy as np
import matplotlib.pyplot as plt

class Gridworld:

    def __init__(self, width, length):
        self.width = width
        self.length = length
        self.obstacle_matrix = np.zeros(shape = (width, length))
        self.stores = []
        self.pe = 0.3
        self.states = []

        # instanciate the state space
        for x in range(width):
            for y in range(length):
                self.states.append([x, y])

    def place_obstacles(self, obstacles):
        for obstacle in obstacles:
            x, y = obstacle
            self.obstacle_matrix[x][y] = 1
    
    def set_icecream_stores(self, coords):
        for coord in coords:
            self.stores.append(coord)

    
    def is_in_statespace(self, state):
        return (0 < state[0] < self.width) and (0 < state[1] < self.length) 
    
    def is_in_obstacles(self, state):
        return bool(self.obstacle_matrix[state[0], state[1]])


    def count_adjacent_states(self, state):
        actions_array = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1]])

        next_states = np.array(state) + actions_array

        adjacent_states = [state for state in next_states if self.is_in_statespace(state) and not self.is_in_obstacles(state)]

        return len(adjacent_states)
    
    def get_adjacent_states(self, state):
        actions_array = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1]])

        next_states = np.array(state) + actions_array
        adjacent_states = [list(state) for state in next_states]
        #adjacent_states = [state for state in next_states if self.is_in_statespace(state) and not self.is_in_obstacles(state)]

        return adjacent_states
    
    def is_adjacent(self, s1, s2):
        return int(np.linalg.norm(np.array(s1) - np.array(s2)) <= 1)

    def harmonic_mean(self, state):
        d1 = np.linalg.norm(np.array(stores[0] - state))
        d2 = np.linalg.norm(np.array(stores[1] - state))

        return 2/(1/d1 + 1/d2)

    def observation(self, state):
        h = self.harmonic_mean(state)
        a = np.ceil(h) - h
        return np.random.choice([np.ceil(h), np.floor(h)], p=[1 - a, a])

    def compute_next_state(state, action):
        a1, a2 = action
        return [state[0] + a1, state[1] + a2]


    def transition(self, state, action):
        adjacent_states = self.get_adjacent_states(state)
        next_state = compute_next_state(state, action)  # add the action to the state
        print('naïve next state:', next_state)
        adjacent_states_if_fail = [state for state in adjacent_states if state != next_state]
        print('undesired states:', adjacent_states_if_fail)
        roll_dice_state = np.random.choice(adjacent_states_if_fail + [next_state], p=[self.pe/4, self.pe/4, self.pe/4, self.pe/4, 1-self.pe])
        print('rolled state', roll_dice_state)
        if is_in_obstacles(roll_dice_state) or not is_in_statespace(roll_dice_state): # if the next state is an obstacle or off world, don't move
            print('state is not in the environment or is an obstacle: not moving')
            return state
        else:
            return roll_dice_state

world = 












    


