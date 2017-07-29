from graph import *
from matrix import *

L = read_graphs("C:/Users/andy9/Documents/Shared/dataset/eu-graphs", 100)
mat_list = [nx.adjacency_matrix(g).todense() for g in L]
adjacency_naive = np.round(matrix_average(mat_list))
# print(adjacency_naive)
np.savetxt("adjacency_naive.csv", adjacency_naive, delimiter=",")
f = open( "C:/Users/andy9/Documents/Shared/dataset/eu-graphs/pos.p", "rb" )
pos = pickle.load(f)
g = nx.from_numpy_matrix(adjacency_naive)
L = None
adjacency_naive = None
f.close()
mat_list = None
nx.draw(g, pos=pos, with_labels=False, node_size=1)
plt.savefig("adjacency_naive.png", format="PNG", dpi=1000)
