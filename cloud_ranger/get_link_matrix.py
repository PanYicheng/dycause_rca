from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest, ChiSquaredTest, MutualInformationTest
import pandas as pd


def build_graph_pc(data, alpha):
    node_num = len(data)
    X = dict()
    variable_types = dict()
    for i in range(node_num):
        X[str(i)] = data[i]
        variable_types[str(i)] = 'c'
    X = pd.DataFrame(X)

    # run the search
    ic_algorithm = IC(RobustRegressionTest, alpha=alpha)
    graph = ic_algorithm.search(X, variable_types)

    N = [[0]*node_num for i in range(node_num)]
    for u, v in graph.edges:
        N[int(u)][int(v)] = 1
    return N
