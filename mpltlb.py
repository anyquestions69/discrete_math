import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

matrix=[]
n=int(input())
rows = []
for i in range(n):
    matrix.append(input().split(''))

    
print(matrix)

P0= matrix

G = nx.DiGraph(np.matrix(P0))
nx.draw(G, with_labels=True, node_size=300, arrows=True)
plt.show()