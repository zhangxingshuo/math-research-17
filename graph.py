import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np
import random
import math

def generate_graph(num_nodes, add_prob=0.5):
    '''
    Creates graph with specified number of nodes and edge density

    @param num_nodes  Number of nodes to create
    @param add_prob   Probability of adding an edge for every possible
                      edge in the graph, edge density

    Credit: https://stackoverflow.com/a/42653330
    '''
    G = nx.Graph()
    # add nodes to graph
    for i in range(num_nodes):
        G.add_node(i)
    # only add new edges
    for node in G.nodes():
        connected = [to for (fr, to) in G.edges(node)]
        unconnected = [n for n in G.nodes() if n not in connected]
        for other_node in unconnected:
            if random.random() < add_prob:
                G.add_edge(node, other_node)
                unconnected.remove(other_node)
    return G

def join_graphs(G1, G2):
    '''
    Combines two graphs, renaming nodes as necessary to preserve separation
    '''
    return nx.disjoint_union(G1, G2)

def remove_edges(G, prob=0.2, func=lambda _:True):
    '''
    For each node, randomly remove an edge with given probability

    @param G     Graph to remove edge from, in place
    @param prob  Probability of removing a given edge
    @param func  Filter function, only remove edges for which function
                 is true, by default always true

    Credit: https://stackoverflow.com/a/42653330
    '''
    # iterate through all of the nodes
    for node in G.nodes():
        if func(node):
            # list of nodes that are connected to current node
            connected = [to for (fr, to) in G.edges(node) if func(to)]
            # if there is a node to remove
            if len(connected):
                # remove random node with probability 
                if random.random() < prob:
                    remove = random.choice(connected)
                    G.remove_edge(node, remove)
                    print "\tedge removed %d -- %d" % (node, remove)

def add_edges(G, prob=0.2, f=lambda _:True):
    '''
    For each node, randomly add an edge with given probability

    @param G     Graph to add edge in, in place
    @param prob  Probability of adding a given edge
    @param func  Filter function, only remove edges for which function
                 is true, by default always true

    Credit: https://stackoverflow.com/a/42653330
    '''
    # iterate through all of the nodes
    for node in G.nodes():
        if f(node):
            # list of nodes that are unconnected to current node
            connected = [to for (fr, to) in G.edges(node)]
            unconnected = [n for n in G.nodes() if n not in connected and f(n)]
            # if there is node that is not connected, i.e. able to add new edge
            if len(unconnected):
                # add random edge with probability
                if random.random() < prob:
                    add = random.choice(unconnected)
                    G.add_edge(node, add)
                    print "\tedge added %d -- %d" % (node, add)

def generate_separated_graphs(num_nodes1, num_nodes2, num_time_steps, directory):
    '''
    Creates a separated, time-evolving network with two components

    @param num_nodes1      Number of nodes in first componenet
    @param num_nodes2      Number of nodes in second componenet
    @param num_time_steps  Number of discrete time intervals
    @param directory       File directory to be written to, in .graphml format
    '''
    # create two disjoint graphs
    M = generate_graph(num_nodes1)
    N = generate_graph(num_nodes2)

    G = join_graphs(M, N)

    for i in range(num_time_steps):
        # preserve separation
        add_edges(G, func=lambda x: x<num_nodes1)
        add_edges(G,func=lambda x: x>=num_nodes1)

        remove_edges(G)

        # write to specified directory in .graphml format, with timestep as label
        nx.write_graphml(G, directory + "/" + str(i) + ".graphml")

def calculate_expected_adjacency(L):
    '''
    Calculate expected adjacency matrix of list of graphs

    Our definition of expectation is averaging elementwise across all adjacency
    matrices, and then for each element rounding up to 1 if greater than or equal
    to 0.5, and otherwise rounding down to 0. In this way, the graph is composed
    of all 1s and 0s and remains an actual adjacency matrix.

    @param L  List of graphs to average
    @return The expected adjacency matrix
    '''
    matrix_list = [np.matrix(nx.adjacency_matrix(g).todense()) for g in L]
    mean = np.mean(matrix_list, axis=0)
    normalize = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
    return normalize(mean)

def get_degree_vector(G):
    '''
    @param G  NetworkX graph
    @return A numpy vector of degrees of nodes
    '''
    laplacian = np.matrix(nx.laplacian_matrix(G).todense())
    return np.diagonal(laplacian)

def calculate_expected_laplacian(L):
    '''
    Calculates the expected Laplacian matrix of the list of graphs

    Our method is to first calculate the expected adjacency matrix, based on the
    definition above. Then, we average the degrees of each node. To ensure integer
    values, we use the floor (lower) and ceiling (upper) to round down and up, and
    in doing so we can obtain two different expected Laplacian matrices.

    Note that the final matrices are not always actual Laplcian matrices, as the degrees
    of each node and their adjacencies do not always align.

    @param L  List of NetworkX graphs to analyze
    @return A tuple of the lower and upper expected Laplacian matrices
            based on our method of expectation
    '''
    A = calculate_expected_adjacency(L)
    diagonal_vectors = [get_degree_vector(g) for g in L]
    expected_diagonal = np.mean(diagonal_vectors, axis=0)
    np_floor = np.vectorize(math.floor)
    np_ceil = np.vectorize(math.ceil)
    return (np.diag(np_floor(expected_diagonal)) - A, 
            np.diag(np_ceil(expected_diagonal)) - A)
