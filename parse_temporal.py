from __future__ import with_statement

import math
import sys
import datetime
import networkx as nx

filename = sys.argv[1]
with open(filename, "r") as f:
    print("Reading data...")
    raw_data = f.read().split("\n")
    edge_list = [tuple([int(e) for e in elem.split()]) for elem in raw_data]
    edge_list = edge_list[:-1]

    times = [e[2] for e in edge_list]
    node_count = max([e[0] for e in edge_list])
    max_time = max(times)
    min_time = min(times)
    n = 1000
    step = (max_time - min_time) / n

    print("%d nodes from %s to %s" % (node_count, 
        datetime.datetime.fromtimestamp(min_time).isoformat(), 
        datetime.datetime.fromtimestamp(max_time).isoformat()))

    count = 1
    for i in xrange(n):
        g = nx.Graph()
        g.add_nodes_from(range(node_count))
        for edge in edge_list:
            fr = edge[0]
            to = edge[1]
            time = edge[2]

            idx = math.floor( (time - min_time) / step )
            if idx == i:
                g.add_edge(fr, to)

        nx.write_graphml(g, "college-graphs/graph" + str(i) + ".graphml")
        print("%d/%d" % (count, n))
        count += 1