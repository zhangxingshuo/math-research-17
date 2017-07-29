import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np
import random
import math
import pickle

from matrix import *

np.set_printoptions(suppress=True)

#####################
### Graph Methods ###
#####################

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

def remove_edges(G, prob=0.2, f=lambda _:True):
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
        if f(node):
            # list of nodes that are connected to current node
            connected = [to for (fr, to) in G.edges(node) if f(to)]
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
            connected = [to for (fr, to) in G.edges(node) if f(to)]
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
        add_edges(G, f=lambda x: x<num_nodes1)
        add_edges(G, f=lambda x: x>=num_nodes1)

        remove_edges(G)

        # write to specified directory in .graphml format, with timestep as label
        nx.write_graphml(G, directory + "/graph" + str(i) + ".graphml")
        nx.draw_networkx(G)
        plt.savefig(directory + "/graph" + str(i) + ".png", format="PNG")
        plt.gcf().clear()

def read_graphs(directory, num_graphs):
    '''
    Reads in graphs from specified directory

    @return A list of NetworkX graphs
    '''
    graphs = []

    for i in range(num_graphs):
        graph_to_add = nx.read_graphml(directory + "/graph" + str(i) + ".graphml")
        graphs.append(graph_to_add)

    return graphs

def save_images(directory, n):
    '''
    Reads n graphs from given directory and saves images as PNG format
    '''
    for i in range(n):
        g = nx.read_graphml(directory + "/graph" + str(i) + ".graphml", node_type=int)
        nx.draw_networkx(g)
        plt.savefig(directory + "/graph" + str(i) + ".png", format="PNG")
        plt.gcf().clear()

def generate_complete_network(directory, num_nodes, num_time_steps):
    '''
    Generates a semi-compliete network with given number of nodes and time steps and
    saves to PNG format
    '''
    for i in xrange(num_time_steps):
        graph_to_add = nx.complete_graph(num_nodes)
        remove_edges(graph_to_add)
        nx.write_graphml(graph_to_add, directory + "/graph" + str(i) + ".graphml")
        nx.draw_networkx(graph_to_add)
        plt.savefig(directory + "/graph" + str(i) + ".png", format="PNG")
        plt.gcf().clear()

def generate_small_network():
    '''
    Generates a network of five individuals over the course of the week. On weekdays, graph
    is sparse and disconnected, while on weekends, graph is almost complete.

    @return A list of NetworkX graphs
    '''
    weekday_graph = nx.Graph()
    weekday_graph.add_edges_from([(1,2), (2,3), (3,4), (4,5), (5,3)])
    weekend_graph = nx.Graph()
    weekend_graph.add_edges_from([(1,2), (2,3), (3,4), (4,5), (5,3), (1,3), (1,4), (2,3), (2,4)])
    return [weekday_graph]*5 + [weekend_graph]*2


########################
### Analysis Methods ###
########################

def laplacian_matrix(graph):
    '''
    @return A scipy sparse matrix representing the Laplacian of the input graph
    '''
    return nx.laplacian_matrix(graph)

def laplacian_dense(graph):
    '''
    @return A dense scipy matrix representing the Laplacian of the input graph
    '''
    return laplacian_matrix(graph).todense()

def expected_degree_matrix(L):
    '''
    @return A diagonal matrix with entries being the expected degree of each node
    '''
    degree_vectors = [np.diagonal(laplacian_dense(g)) for g in L]
    avg_degree = np.mean(degree_vectors, axis=0)
    return np.diag(avg_degree)

def expected_rotation(L):
    '''
    @return The expected rotation matrix using logarithmic mean
    '''
    matrix_list = [eigvech(laplacian_matrix(g)) for g in L]
    return log_matrix_average(matrix_list)

def expected_rotation_polar(L):
    '''
    Given a list of graphs, calculates the spectral decomposition of each Laplacian. For
    each matrix M of eigenvectors, polar decompose into unitary matrix U and Hermitian
    matrix P. 

    Sum the unitary matrices and polar decompose the sum. The unitary matrix from the
    decomposition is the expected rotation.
    '''
    matrix_list = [eigvech(laplacian_matrix(g)) for g in L]
    return polar_decomp_average(matrix_list)

def expected_eigenvalues(L):
    '''
    @param L  List of graphs to analyze
    @return A vector representing the expected Laplacian spectrum for the graphs
    '''
    eigen_list = [eigvalsh(laplacian_matrix(g)) for g in L]
    return np.mean(eigen_list, axis=0)

def expected_laplacian(L):
    '''
    Calculate the expected Laplacian of a time-varying network using

    L_e = P_e * G_e * (P_e)^T

    where P_e is the expected rotation matrix and G_e is a diagonal matrix
    of eigenvalues.

    @param L A list of NetworkX graphs
    @return A numpy matrix representing the expected Laplcian of the network
    '''
    P = expected_rotation_polar(L)
    D = np.diag(expected_eigenvalues(L))
    return np.dot(np.dot(P, D), P.T)

if __name__ == "__main__":
    # print("Reading data...")
    L = read_graphs("C:/Users/andy9/Documents/Shared/dataset/eu-graphs", 100)
    # print("Calculating...")
    # laplacian = expected_laplacian(L)
    # degree_matrix = expected_degree_matrix(L)
    # # print(laplacian)
    # np.savetxt("laplacian.csv", laplacian, delimiter=",")
    # # print(degree_matrix)
    # adjacency = degree_matrix - laplacian
    # degree_matrix = None
    # laplacian = None
    # # print(adjacency)
    # np.fill_diagonal(adjacency, 0)
    # adjacency = np.round(adjacency)
    # np.savetxt("adjacency_polar.csv", adjacency, delimiter=",")

    # f = open( "C:/Users/andy9/Documents/Shared/dataset/stack-graphs/pos.p", "rb" )
    # pos = pickle.load(f)
    # g = nx.from_numpy_matrix(adjacency)
    # nx.draw(g, pos=pos, with_labels=False, node_size=1)
    # plt.savefig("adjacency_polar.png", format="PNG", dpi=1000)
    # DELETE L = generate_small_network()
    adjacency_list = [nx.adjacency_matrix(g).todense() for g in L]
    vector_tuple = tuple([mat.flatten().T for mat in adjacency_list])
    X = np.hstack(vector_tuple)
    sigma = np.cov(X) # TODO: replace sigma_i,j = E[(X_i)(X_j)] - (mu_i)(mu_j)
    # NOTE: for np vector, getting i element is vec[i][0,0]