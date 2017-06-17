import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np
import random

# nx.write_graphml(G, "graphs/1.graphml")

# G = nx.read_graphml("graphs/1.graphml", node_type=int)

def generate_graph(num_nodes, add_prob=0.5):
    '''
    Creates graph with specified number of nodes and edge density
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
    return nx.disjoint_union(G1, G2)

def remove_edges(G, prob=0.2, f=lambda x:True):
    for node in G.nodes():
        if f(node):
            connected = [to for (fr, to) in G.edges(node) if f(to)]
            if len(connected):
                if random.random() < prob:
                    remove = random.choice(connected)
                    G.remove_edge(node, remove)
                    print "\tedge removed %d -- %d" % (node, remove)

def add_edges(G, prob=0.2, f=lambda x:True):
    for node in G.nodes():
        if f(node):
            connected = [to for (fr, to) in G.edges(node)]
            unconnected = [n for n in G.nodes() if n not in connected and f(n)]

            if len(unconnected):
                if random.random() < prob:
                    add = random.choice(unconnected)
                    G.add_edge(node, add)
                    print "\tedge added %d -- %d" % (node, add)

# create two disjoint graphs
M = generate_graph(50)
N = generate_graph(50)

G = join_graphs(M, N)

for i in range(50):
    add_edges(G, f=lambda x: x<50)
    add_edges(G,f=lambda x: x>=50)
    remove_edges(G)
    nx.write_graphml(G, "graphs/" + str(i) + ".graphml")

# nx.draw_networkx(G)
# plt.show()

# nx.write_graphml(G, "graph.graphml")
# L = nx.read_graphml("graph.graphml")

# mat = nx.adjacency_matrix(G)
# print np.matrix(mat.todense())

# lp = nx.laplacian_matrix(G)
# print np.matrix(lp.todense())
