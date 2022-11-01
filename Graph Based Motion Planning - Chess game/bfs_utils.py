import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Graph:

    def __init__(self, width, length):
        self.edges = []
        self.state_space = []
        self.nodes = []
        self.width = width  # width of the graph
        self.length = length  # length of the graph

        # Initialize the graph
        for x in range(self.width):
            for y in range(self.length):
                self.add_node([x,y])
        for node in self.nodes:
            edges = self.get_edges(node)
            self.add_edges(edges)

    def add_node(self, coords):
        self.nodes.append(coords)


    def add_edges(self, list_of_edges):
        self.edges += list_of_edges
    
    def add_edge(self, node1, node2):
        self.edges.append((node1, node2))

    def get_edges(self, node):  # get the edges leaving that node
        actions_array = np.array([[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,1],[-2,-1]])

        result = np.array(node) + actions_array

        result_new = np.array([element for element in result if 0 <= element[0] < self.width and 0 <= element[1] < self.length])
        #node_array = np.broadcast_to(node, np.shape(result_new)[0])

        r = [(node, adj_node) for adj_node in result_new]

        return r
    
    
    def get_neighbors(self, node):
        actions_array = np.array([[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,1],[-2,-1]])

        result = np.array(node) + actions_array
        result_new = [list(element) for element in result if 0 <= element[0] < self.width and 0 <= element[1] < self.length]

        return result_new



# Create the graph and all the nodes

width = 8
length = 8
chess_board = Graph(width, length)



'''
for x in range(width):
    for y in range(length):
        chess_board.add_node([x,y])

# Add all the edges

for node in chess_board.nodes:
    edges = chess_board.get_edges(node)
    chess_board.add_edges(edges)

'''

def naive_breadth_first_search(graph, init_node):

    # Retrieves all the nodes in the graph
    list_nodes = [init_node]
    k = 0
    while k < len(list_nodes):
        node = list_nodes[k]
        next_nodes = graph.get_neighbors(node)
        list_nodes += [element for element in next_nodes if element not in list_nodes]
        k += 1
    return list_nodes


retrieved_nodes = naive_breadth_first_search(chess_board, init_node = [3,6])
print("The correct number of nodes is retrieved:", len(retrieved_nodes) == len(chess_board.nodes))


# Now instead of retrieving the nodes, let us retrieve the optimal paths

def breadth_first_search(graph, init_node, goal_node):
    path_list = [[init_node]]
    k = 0
    while k < len(path_list) and k < 1000:

        path = path_list[k]
        last_node = path[-1]
        neighbors = graph.get_neighbors(last_node)
        for neighbor in neighbors:
            if neighbor == goal_node:
                return path + [neighbor]
            else:
                if neighbor not in path and path + [neighbor] not in path_list:
                    path_list.append(path + [neighbor])
        k += 1
    
    print("no path detected")
    return path_list





# Create the graph and all the nodes

width = 8
length = 8
chess_board = Graph(width, length)



init_node = [0, 0]
goal_node = [6, 0]
path_bfs = breadth_first_search(chess_board, init_node, goal_node)


width = 4
length = 4
chess_board = Graph(width, length)

# Let us now implement depth first search



def naive_depth_first_search(chess_board, init_node):  # just explores the nodes
    node_list = []
    queue = [init_node]

    while queue:
        node = queue.pop(0)
        node_list.append(node)
        neighbors = chess_board.get_neighbors(node)
        neighbors = [neighbor for neighbor in neighbors if neighbor not in queue + node_list]
        if neighbors != []:
            queue =  neighbors + queue
    
    return node_list


# Let's try it !
init_node = [0, 0]
node_list = naive_depth_first_search(chess_board, init_node)
print("The correct number of nodes is retrieved:", len(node_list) == len(chess_board.nodes))


def depth_first_search(chess_board, init_node, goal_node):  # just explores the nodes
    path_list = []
    path_queue = [[init_node]]
    while path_queue:
        path = path_queue.pop(0)
        #print(path)
        node = path[-1]
        
        neighbors = chess_board.get_neighbors(node)
        #print('neighbors:', neighbors)
        for neighbor in neighbors:
            if neighbor == goal_node:
                return path + [neighbor]
            else:
                #print('neighbor', neighbor)
                #print('path + neighbor', path + [neighbor])
                if neighbor not in path and path + [neighbor] not in path_list:
                    new_path = path + [neighbor]
                    #print('new_path', new_path)
                    path_queue = [new_path] + path_queue
                    #print('path_queue', path_queue)
                    path_list.append(new_path)

    print('No path found')
    return path_list




# Create the graph and all the nodes

width = 8
length = 8
chess_board = Graph(width, length)


init_node = [0, 0]
goal_node = [6, 0]
path_dfs = depth_first_search(chess_board, init_node, goal_node)

print('Length of the path found with breadth_first_search:', len(path_bfs))
print('Length of the path found with depth_first_search:', len(path_dfs))


# We can see that DFS definitely does not give you the shortest path...








