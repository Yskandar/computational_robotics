import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Graph:

    def __init__(self, width, length):
        self.edges = []
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

# Create the graph and all the nodes

width = 8
length = 8
chess_board = Graph(width, length)

for i in range(width):
    for j in range(length):
        chess_board.add_node(np.array([i,j]))

# Add all the edges

for node in chess_board.nodes:
    edges = chess_board.get_edges(node)
    chess_board.add_edges(edges)












