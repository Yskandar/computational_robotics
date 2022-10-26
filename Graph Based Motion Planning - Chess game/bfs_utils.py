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

for x in range(width):
    for y in range(length):
        chess_board.add_node([x,y])

# Add all the edges

for node in chess_board.nodes:
    edges = chess_board.get_edges(node)
    chess_board.add_edges(edges)



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
                    print(path_list)
        k += 1
    
    print("no path detected")
    return path_list





# Create the graph and all the nodes

width = 3
length = 3
chess_board = Graph(width, length)



init_node = [0, 0]
goal_node = [10, 0]
paths = breadth_first_search(chess_board, init_node, goal_node)










