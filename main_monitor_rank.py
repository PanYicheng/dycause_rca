import os
import random


import numpy as np
import pandas as pd

from cloud_ranger.get_link_matrix import build_graph_pc
from util_funcs.draw_graph import draw_weighted_graph
from util_funcs.evaluation_function import prCal, my_acc, print_prk_acc
from util_funcs.excel_utils import readExl, saveToExcel
from util_funcs.pearson import calc_pearson
from util_funcs.loaddata import load, aggregate


def firstorder_randomwalk(
    P,
    epochs,
    start_node,
    teleportation_prob,
    label=[],
    walk_step=1000,
    print_trace=False,
):
    n = P.shape[0]
    score = np.zeros([n])
    current = start_node - 1
    for epoch in range(epochs):
        if print_trace:
            print("\n{:2d}".format(current + 1), end="->")
        for step in range(walk_step):
            if np.sum(P[current]) == 0:
                current = np.random.choice(range(n), p=teleportation_prob)
                break
            else:
                next_node = np.random.choice(range(n), p=P[current])
                if print_trace:
                    print("{:2d}".format(current + 1), end="->")
                score[next_node] += 1
                current = next_node
    score_list = list(zip(label, score))
    score_list.sort(key=lambda x: x[1], reverse=True)
    return score_list


def normalize(p):
    """Normalize the matrix in each row
    """
    p = p.copy()
    for i in range(p.shape[0]):
        row_sum = np.sum(p[i])
        if row_sum > 0:
            p[i] /= row_sum
    return p


def relaToRank(rela, access, rankPaces, frontend, rho=0.3, print_trace=False):
    n = len(access)
    S = [abs(_) for _ in rela[frontend - 1]]
    P = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            # forward edge
            if access[i][j] != 0:
                P[i, j] = abs(S[j])
            # backward edge
            elif access[j][i] != 0:
                P[i, j] = rho * abs(S[i])
    # Add self edges
    for i in range(n):
        if i != frontend - 1:
            P[i][i] = max(0, S[i] - max(P[i]))
    P = normalize(P)

    teleportation_prob = (np.array(S) / np.sum(S)).tolist()
    label = [i for i in range(1, n + 1)]
    l = firstorder_randomwalk(
        P, rankPaces, frontend, teleportation_prob, label, 
        print_trace=print_trace
    )
    # print(l)
    return l, P


def test_monitor_rank(
    data_source="real_micro_service",
    pc_aggregate=5,
    pc_alpha=0.1,
    testrun_round=1,
    frontend=14,
    true_root_cause=[6, 28, 30, 31],
    rho=0.2,
    save_data_fig=False,
    verbose=False,
    runtime_debug=True,
    *args,
    **kws
):
    """
    Params:
        save_data_fig: whether save transition matrix and the graph
        runtime_debug: whether enable runtime debug mode, where each process is always executed.
        verbose: the debugging print level: 0 (Nothing), 1 (Method info), 2 (Phase info), 3(Algorithm info)
    """
    np.random.seed(42)
    random.seed(42)

    if verbose:
        # verbose level >= 1: print method name
        print("{:#^80}".format("Monitor Rank"))
        if verbose>=3:
            # verbose level >= 3: print method parameters
            print("{:-^80}".format(data_source))
            print("{:^10}pc aggregate  :{}".format("", pc_aggregate))
            print("{:^10}pc alpha      :{}".format("", pc_alpha))
            print("{:^10}rho           :{}".format("", rho))
    # Load data
    # Use raw_data, data_head if it is provided in kws
    if 'data' not in kws:
        data, data_head = load(
            os.path.join("data", data_source, "rawdata.xlsx"),
            normalize=True,
            zero_fill_method='prevlatter',
            aggre_delta=pc_aggregate,
            verbose=verbose,
        )
        # Transpose data to shape [N, T]
        data = data.T
    else:
        data_head = kws['data_head']
        # raw_data is of shape [T, N]. Here we first transpose it to [N, T] and then aggregate.
        raw_data = kws['data']
        data = np.array([aggregate(row, pc_aggregate) for row in raw_data.T])

    rela = calc_pearson(data, method="numpy", zero_diag=False)
    
    # region Build call graph from file or PC algorithm or parameters in kws
    
    # The file name for saving dependency graph
    window_start=0
    if 'window_start' in kws:
        window_start=kws['window_start']
    dep_graph_filepath = os.path.join(
        "monitor_rank",
        "results",
        data_source,
        "dep_graph_agg{}_alpha{}_winstart{}_len{}.xlsx".format(pc_aggregate, pc_alpha, window_start, data.shape[1]),
    )
    
    # When PC dep_graph isn't given, use PC algorithm
    if 'dep_graph' not in kws:
        if data_source == "pymicro":
            # Real call topology matrix
            dep_graph = readExl(os.path.join("data", data_source, "true_callgraph.xlsx"))
        elif data_source == "real_micro_service":
            # If it is not in runtime_debug mode, save and load previous constructed graph if possible
            if os.path.exists(dep_graph_filepath) and not runtime_debug:
                # If previous dependency graph exists, load it.
                if verbose and verbose >= 2:
                    # verbose level >= 2: print dependency graph loading info
                    print(
                        "{:^10}Loading existing link matrix file: {}".format(
                            "", dep_graph_filepath
                        )
                    )
                dep_graph = readExl(dep_graph_filepath)
            else:
                # If previous dependency graph doesn't exist, genereate it using PC algorithm.
                if verbose and verbose >= 2:
                    # verbose level >= 2: print dependency graph construction info
                    print("{:^10}Generating new link matrix".format(""))
                dep_graph = build_graph_pc(data, alpha=pc_alpha)
    # When PC dep_graph is given, use dep_graph given
    else:
        dep_graph = kws['dep_graph']
    # If not in runtime debugging mode, cache dependency graph
    if not runtime_debug:
        os.makedirs(os.path.dirname(dep_graph_filepath), exist_ok=True)
        saveToExcel(dep_graph_filepath, dep_graph)
    callgraph = dep_graph
    # endregion
    
    topk_list = range(1, 6)
    prkS = [0] * len(topk_list)
    acc = 0
    for i in range(testrun_round):
        if verbose and verbose >= 3:
            # verbose level >= 3: print random walk starting info
            print("{:^15}Randwalk round:{}".format("", i))
            print(
                "{:^15}Starting randwalk at({}): {}".format(
                    "", frontend, data_head[frontend - 1]
                )
            )
        rank, P = relaToRank(rela, callgraph, 10, frontend, rho=rho, print_trace=False)
        acc += my_acc(rank, true_root_cause, n=len(data))
        for j, k in enumerate(topk_list):
            prkS[j] += prCal(rank, k, true_root_cause)

        if verbose and verbose > 1:
            # verbose level >= 2: print random walk rank results
            print("{:^15}".format(""), end="")
            for j in range(len(rank)):
                print(rank[j], end=", ")
            print("")

    for j, k in enumerate(topk_list):
        prkS[j] = float(prkS[j]) / testrun_round
    acc /= testrun_round
    # Display PR@k and Acc if disable_print is not set
    if 'disable_print' not in kws or kws['disable_print'] is False:
        print_prk_acc(prkS, acc)
    if save_data_fig:
        saveToExcel(
            os.path.join(
                "monitor_rank",
                "results",
                data_source,
                "transition_prob_ela{}.xlsx".format(pc_aggregate),
            ),
            P.tolist(),
        )
        draw_weighted_graph(
            P.tolist(),
            os.path.join(
                "monitor_rank",
                "results",
                data_source,
                "transition_graph_ela{}.png".format(pc_aggregate),
            ),
        )
    return prkS, acc
