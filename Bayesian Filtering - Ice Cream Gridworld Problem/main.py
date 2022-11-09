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

        # compute the harmonic means
        self.compute_harmonic_means()

    def place_obstacles(self, obstacles):
        for obstacle in obstacles:
            x, y = obstacle
            self.obstacle_matrix[x][y] = 1
    
    def place_icecream_stores(self, coords):
        for coord in coords:
            self.stores.append(coord)

    
    def is_in_statespace(self, state):
        return (0 <= state[0] < self.width) and (0 <= state[1] < self.length) 
    
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

    def compute_next_state(self, state, action):
        a1, a2 = action
        return [state[0] + a1, state[1] + a2]


    def transition(self, state, action):
        actions = [[0,0],[1,0],[0,1],[-1,0],[0,-1]]
        if action not in actions:
            print('invalid action: did not move')
            return state


        adjacent_states = self.get_adjacent_states(state)
        next_state = self.compute_next_state(state, action)  # add the action to the state
        print('naïve next state:', next_state)
        adjacent_states_if_fail = [state for state in adjacent_states if state != next_state]
        print('undesired states:', adjacent_states_if_fail)
        roll_dice_state_idx = np.random.choice([0, 1, 2, 3, 4], p=[self.pe/4, self.pe/4, self.pe/4, self.pe/4, 1-self.pe])
        roll_dice_state = (adjacent_states_if_fail + [next_state])[roll_dice_state_idx]
        print('rolled state', roll_dice_state)
        print('rolled state is desired state: ', roll_dice_state == next_state)
        if self.is_in_obstacles(roll_dice_state) or not self.is_in_statespace(roll_dice_state): # if the next state is an obstacle or off world, don't move
            print('state is not in the environment or is an obstacle: not moving')
            return state
        else:
            print('Moving to rolled state')
            return roll_dice_state

    def compute_harmonic_means(self):
        # harmonic mean for all the states in the belief state

        d1 = np.linalg.norm(np.array(self.stores[0]) - np.array(self.states), axis = 1)
        d2 = np.linalg.norm(np.array(self.stores[1]) - np.array(self.states), axis = 1)
        
        self.harmonic_means = 2/((1/d1) + (1/d2))

    def compute_posteriors(self, observation):
        """
        Computes the posterior probabilities pr(o|s) for all the states
        """

        posteriors = np.empty(shape = self.harmonic_means.shape)
        floor = np.floor(self.harmonic_means)
        ceil = np.ceil(self.harmonic_means)

        a = np.where(floor == observation, 1, 0) * (1 - (ceil - self.harmonic_means))
        b = np.where(ceil == observation, 1, 0) * (ceil - self.harmonic_means)

        return a + b

    def observation_update(self, belief_state, observation):
        
        posteriors = self.compute_posteriors(observation)
        return belief_state * posteriors

    def compute_probabilities(self, action):
        state_matrix_1 = np.broadcast_to(np.array(self.states), shape = (len(self.states), len(self.states)))
        state_matrix_2 = np.broadcast_to(np.array(self.states).T, shape = (len(self.states), len(self.states)))
        probabilities = np.empty(shape = (len(self.states), len(self.states)))

        a = np.where(state_matrix == self.compute_next_state(state_matrix, action) 
                    and self.is_in_statespace(state_matrix) and not self.is_in_obstacles(state_matrix), 1 - self.pe, 0)

    

    




world = Gridworld(5, 5)
# put the obstacles
obstacles = [[1, 1], [2, 1], [1,3], [2, 3]]
world.place_obstacles(obstacles)

# put the stores
stores = [[2, 0], [2, 2]]
world.place_icecream_stores(stores)

# initial state
initial_state = [0, 0]
action = [-1, 0]
world.is_in_statespace([0,1])
print(world.transition(initial_state, action))



# Belief state

initial_belief = np.random.rand(5, 5)


















    


