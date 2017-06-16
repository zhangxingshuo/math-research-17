import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np

G = nx.Graph()
for i in xrange(1,10):
    G.add_edge(i-1, i)

mat = nx.adjacency_matrix(G)
print np.matrix(mat.todense())

lp = nx.laplacian_matrix(G)
print np.matrix(lp.todense())

nx.draw_networkx(G)
plt.show()