import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
G = nx.karate_club_graph()
comp = girvan_newman(G)
print(tuple(sorted(c) for c in next(comp)))
nx.draw_circular(G, with_labels=True)
plt.show()