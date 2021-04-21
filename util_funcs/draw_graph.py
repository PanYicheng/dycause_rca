import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def draw_weighted_graph(transition_matrix, filename, weight_multiplier=4):
    """Draw weighted graph given transition probability
    
    The parameter is a transition matrix which the i,j element
    represent the probability of transition from i to j
    The weight is illustrated using edge width
    """
    n = len(transition_matrix)
    fig, ax = plt.subplots(figsize=(20, 20))
    g = nx.MultiDiGraph()
    edge = []
    edge_weight = dict()
    pos = dict()
    for i in range(1, n+1):
        angle = (i+0.0)/n*(2*np.pi)
        pos[i] = (np.cos(angle), np.sin(angle))
    g.clear()
    ax.clear()
    for x_i in range(1, n+1):
        for y_i in range(1, n+1):
            if transition_matrix[x_i-1][y_i-1] > 0:
                g.add_edge(x_i, y_i)
                edge.append((x_i, y_i))
                edge_weight[(x_i, y_i)] = transition_matrix[x_i-1][y_i-1]
    # No edge, just return
    if len(edge) == 0:
        return
    nx.draw_networkx(g, pos, with_labels=True, ax=ax, node_size=1000, font_size=25,
                     font_color='y', node_color='k', arrows=False)
    weight_list = np.array([edge_weight[key] for key in g.edges()])
    weight_list = weight_list / np.max(weight_list) * weight_multiplier
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=g.edges(), alpha=0.6,
                           width=weight_list, arrowsize=35,
                           connectionstyle='arc3,rad=0')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Create directory if not exists
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.savefig(filename)
    # plt.show()
