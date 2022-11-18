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
        #self.compute_harmonic_means()

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
    
    def get_adjacent_states(self, state, mode = 'transition'):
        actions_array = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1]])

        next_states = np.array(state) + actions_array
        adjacent_states = [list(state) for state in next_states]
        if mode == 'probabilities':
            adjacent_states = [next_state for next_state in next_states if self.is_in_statespace(next_state) and not self.is_in_obstacles(next_state) and np.any(next_state != state)]
        return adjacent_states
    
    def is_adjacent(self, s1, s2):
        return int(np.linalg.norm(np.array(s1) - np.array(s2)) <= 1)

    def harmonic_mean(self, state):
        d1 = np.linalg.norm(np.array(self.stores[0]) - np.array(state))
        d2 = np.linalg.norm(np.array(self.stores[1]) - np.array(state))

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
        if not self.is_in_statespace(roll_dice_state) or self.is_in_obstacles(roll_dice_state) : # if the next state is an obstacle or off world, don't move
            print('state is not in the environment or is an obstacle: not moving')
            return state
        else:
            print('Moving to rolled state')
            return roll_dice_state


    def transitionProbabilities(self, state, next_state, action):
        theoretical_next_state = self.compute_next_state(state,action)
        num_adjacent_states = len(self.get_adjacent_states(state, mode="probabilities"))
        if next_state == theoretical_next_state and self.is_in_statespace(state) and not self.is_in_obstacles(state) and next_state != state:
            return 1 - self.pe

        elif next_state != theoretical_next_state and self.is_in_statespace(state) and not self.is_in_obstacles(state) and next_state != state: 
            return self.is_adjacent(next_state,state) * self.pe/4

        elif next_state != theoretical_next_state and self.is_in_statespace(state) and not self.is_in_obstacles(state) and next_state == state:
            return self.is_adjacent(next_state,state) * (self.pe - (num_adjacent_states * self.pe/4) )
        
        elif next_state == theoretical_next_state == state:
            return 1
        
        else:
            return 0


    """
    def probabilities(self, state, next_state, action):
        
		probability = 0
		# States must not be occupied by obstacle and they must be adjacent
		if not self.is_in_obstacles(next_state) and not self.is_in_obstacles(state) and self.is_adjacent(next_state, state):
			# If desired action succeeded
			if next_state == self.compute_next_state(state, action):
				# If the state changes
				if state == next_state:
					probability = 1
				else:
					probability = 1 - self.pe
			# If desired action failed
			else:
				# If the state changes
				if state == next_state:
					probability = 1
					# Subtract the probability of all surrounding states
					for s in self.get_adjacent_states(state, mode = "probabilities"):
						if s != state:
							probability -= self.probabilities(state, action, s)
				else:
					if action != [0, 0]:
						probability = self.pe/4.0
        return probability
    """


    def compute_harmonic_means(self):

        # harmonic mean for all the states in the belief state
        d1 = np.linalg.norm(np.array(self.stores[0]) - np.array(self.states), axis = 1)
        d2 = np.linalg.norm(np.array(self.stores[1]) - np.array(self.states), axis = 1)
        self.harmonic_means = np.reshape(2/((1/d1) + (1/d2)), (1, len(self.states)))

    def compute_posteriors(self, observation):
        """
        Computes the posterior probabilities pr(o|s) for all the states
        """

        posteriors = np.empty(shape = self.harmonic_means.shape)
        floor = np.floor(self.harmonic_means)
        ceil = np.ceil(self.harmonic_means)

        a = np.where(ceil == observation, 1, 0) * (1 - (ceil - self.harmonic_means))
        b = np.where(floor == observation, 1, 0) * (ceil - self.harmonic_means)

        return a + b

    def observation_update(self, belief_state, observation):
        
        posteriors = self.compute_posteriors(observation)
        return (belief_state * posteriors) / np.sum(belief_state * posteriors)

    def compute_probabilities(self, action):
        probabilities = np.empty(shape = (len(self.states), len(self.states)))

        for i in range(len(self.states)):
            for j in range(len(self.states)):
                current_state = self.states[i]
                next_state = self.states[j]
                probabilities[i, j] = self.transitionProbabilities(current_state, next_state, action)
        
        return probabilities

    def dynamics_update(self, belief_state, action):
        probabilities_matrix = self.compute_probabilities(action)

        return (belief_state @ probabilities_matrix) / np.sum(belief_state @ probabilities_matrix)




world = Gridworld(5, 5)
# put the obstacles
obstacles = [[1, 1], [2, 1], [1,3], [2, 3]]
world.place_obstacles(obstacles)

# put the stores
stores = [[2, 0], [2, 2]]
world.place_icecream_stores(stores)

# Compute harmonic means
world.compute_harmonic_means()

current_state = [0, 0]
p = 1/len(world.states)
current_belief = np.ones((1,len(world.states))) * p
for i in range(100):
    # initial state
    action = [1, 0]
    world.is_in_statespace([0,1])
    current_state = world.transition(current_state, action)
    print("current_state", current_state)



    # take the action
    probabilities_matrix = world.compute_probabilities(action)
    #print(probabilities_matrix)

    # update the belief_state
    current_belief = world.dynamics_update(current_belief, action)

    # Get new observation
    obs = world.observation(current_state)
    print("observation", obs)
    # update the belief_state
    current_belief = world.observation_update(current_belief, obs)
    print("current_belief", current_belief)

print(world.states)

















    


