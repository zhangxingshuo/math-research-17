import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np
import random
import math
import pickle
import scipy.stats
import csv

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
        try:
            graph_to_add = nx.read_graphml(directory + "/graph" + str(i) + ".graphml")
            graphs.append(graph_to_add)
        except:
            print("Error at index: " + str(i))

    return graphs

def save_images(directory, n):
    '''
    Reads n graphs from given directory and saves images as PNG format
    '''
    pos = None
    for i in range(n):
        g = nx.read_graphml(directory + "/graph" + str(i) + ".graphml", node_type=int)
        if i == 0:
            pos = nx.spring_layout(g)
        nx.draw(g, pos=pos, with_labels=False, node_size=1)
        plt.savefig(directory + "/graph" + str(i) + ".png", format="PNG")
        plt.gcf().clear()
        print(i)
    return pos

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

def covariance(i, j, samples, mean):
    '''
    Calculates single covariance between ith and jth elements in
    sample vectors, given a mean vector
    '''
    # calculate expected value X_i*X_j
    total = 0
    for sample in samples:
        total += sample[i][0,0]*sample[j][0,0]
    total /= len(samples)

    return total - mean[i][0,0]*mean[j][0,0]

def total_covariance(L):
    # adjacency_list = [nx.adjacency_matrix(g).todense() for g in L]
    # vectors = [mat.flatten().T for mat in adjacency_list]
    # mean = matrix_average(vectors)
    # n = len(mean)
    # print("Total: %d" % n)
    # X = np.hstack(tuple(vectors))
    # sigma = np.cov(X) 
    # with open("output.csv", "w") as f:
    #     for i in range(n):
    #         f.write(','.join([str(covariance(i, j, vectors, mean)) for j in range(n)]) + '\n')
    #         print("%d/%d" & (i, n))
    # cov = {}
    # for i in range(n):
    #     for j in range(n):
    #         if (i, j) not in cov and (j, i) not in cov:
    #             cov[(i,j)] = covariance(i, j, vectors, mean)
    #     print("%d/%d" % (i+1, n))
    # pickle.dump(cov, open("cov.pkl", "wb"))
    return

def normal_divergence(G):
    degree_count = nx.degree(G).values()
    num_nodes = len(degree_count)
    max_degree = max(degree_count)

    mean = sum(degree_count) / num_nodes
    variance = np.var(degree_count)
    normal = scipy.stats.norm(mean, math.sqrt(variance))
    print mean, variance

    degree_dist = [1.0 * degree_count.count(i) / num_nodes for i in range(max_degree + 1)]

    print degree_dist
    cutoff = int(round(mean + 8 * math.sqrt(variance)))
    degree_dist = degree_dist[:cutoff]
    print degree_dist

    for i in range(len(degree_dist)):
        if degree_dist[i] == 0:
            degree_dist[i] += 0.01
            degree_dist[np.argmax(degree_dist)] -= 0.01

    mean = sum(degree_count) / num_nodes
    variance = np.var(degree_count)
    normal = scipy.stats.norm(mean, math.sqrt(variance))
    
    normal_dist = [normal.cdf(i + 1) - normal.cdf(i) for i in range(len(degree_dist))]

    print "Degree" 
    print degree_dist
    print "Normal"
    print normal_dist
    
    div1 = scipy.stats.entropy(degree_dist, normal_dist)
    div2 = scipy.stats.entropy(normal_dist, degree_dist)

    return (div1 + div2) / 2

if __name__ == "__main__":
    # Read data
    print("Reading data...")
    L = read_graphs("C:/Users/andy9/Documents/Shared/dataset/college-graphs", 100)

#     # Perform expected value calculations
#     print("Calculating...") 
#     laplacian = expected_laplacian(L)
#     degree_matrix = expected_degree_matrix(L)
#     print(laplacian)
#     np.savetxt("laplacian.csv", laplacian, delimiter=",")
#     print(degree_matrix)
#     adjacency = degree_matrix - laplacian

#     # Free up memory explicitly
#     degree_matrix = None
#     laplacian = None
#     print(adjacency)
#     np.fill_diagonal(adjacency, 0)
#     adjacency = np.round(adjacency)
#     np.savetxt("adjacency_polar.csv", adjacency, delimiter=",")

    # Draw graph
    # f = open( "C:/Users/andy9/Documents/Shared/dataset/college-graphs/pos.p", "rb" )
    # pos = pickle.load(f)
    # g = nx.from_numpy_matrix(adjacency)
    # nx.draw(g, pos=pos, with_labels=False, node_size=1)
    # nx.draw_networkxaw(g, with_labels=False, node_size=1)
    # plt.savefig("adjacency_polar.png", format="PNG", dpi=1000)
    save_images("C:/Users/andy9/Documents/Shared/dataset/college-graphs", 100)

    divergence_list = []
    for i in range(len(L)):
        print i
        graph = L[i]
        divergence_list.append(normal_divergence(graph))

    with open("divergence.csv", "wb") as file:
        wr = csv.writer(file)
        for elem in divergence_list:
            wr.writerow([elem])