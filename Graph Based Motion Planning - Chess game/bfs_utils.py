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

    def add_node(self, coords):
        self.nodes.append(coords)


    def add_edges(self, list_of_edges):
        self.edges += list_of_edges
    
    def add_edge(self, node1, node2):
        self.edges.append((node1, node2))

    def get_edges(self, node):  # get the edges leaving that node
        actions_array = np.array([[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,1],[-2,-1]])

        result = node + actions_array

        result_new = np.array([element for element in result if 0 <= element[0] < self.width and 0 <= element[1] < self.length])
        #node_array = np.broadcast_to(node, np.shape(result_new)[0])

        r = [(node, adj_node) for adj_node in result_new]

        return r
    
    def get_neighbors(self, node):
        actions_array = np.array([[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,1],[-2,-1]])

        result = node + actions_array
        result_new = [element for element in result if 0 <= element[0] < self.width and 0 <= element[1] < self.length]

        return result_new



# Create the graph and all the nodes

width = 8
length = 8
chess_board = Graph(width, length)

for x in range(width):
    for y in range(length):
        chess_board.add_node(np.array([x,y]))

# Add all the edges

for node in chess_board.nodes:
    edges = chess_board.get_edges(node)
    chess_board.add_edges(edges)



def naive_breadth_first_search(graph, init_node):
    list_nodes = [list(init_node)]
    k = 0
    while k <= len(list_nodes):
        node = np.array(list_nodes[k])
        next_nodes = graph.get_neighbors(node)
        list_nodes += [list(element) for element in next_nodes if list(element) not in list_nodes]
        k += 1
    return list_nodes



test = naive_breadth_first_search(chess_board, init_node = np.array([0,0]))









