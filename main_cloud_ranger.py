import os
import random
import time


import numpy as np
import pandas as pd

from cloud_ranger.get_link_matrix import build_graph_pc
from util_funcs.draw_graph import draw_weighted_graph
from util_funcs.evaluation_function import prCal, my_acc, print_prk_acc
from util_funcs.excel_utils import readExl, saveToExcel
from util_funcs.pearson import calc_pearson
from util_funcs.loaddata import load, aggregate


def secondorder_randomwalk(
    M, epochs, start_node, label=[], walk_step=1000, print_trace=False
):
    n = M.shape[0]
    score = np.zeros([n])
    for epoch in range(epochs):
        previous = start_node - 1
        current = start_node - 1
        if print_trace:
            print("\n{:2d}".format(current + 1), end="->")
        for step in range(walk_step):
            if np.sum(M[previous, current]) == 0:
                break
            next_node = np.random.choice(range(n), p=M[previous, current])
            if print_trace:
                print("{:2d}".format(current + 1), end="->")
            score[next_node] += 1
            previous = current
            current = next_node
    score_list = list(zip(label, score))
    score_list.sort(key=lambda x: x[1], reverse=True)
    return score_list


def guiyi(p):
    """Normalize matrix column-wise.
    """
    nextp = [[0 for i in range(len(p[0]))] for j in range(len(p))]
    for i in range(len(p)):
        for j in range(len(p[0])):
            lineSum = (np.sum(p, axis=1))[i]
            if lineSum == 0:
                break
            nextp[i][j] = p[i][j] / lineSum
    return nextp


def relaToRank(rela, access, rankPaces, frontend, beta=0.1, rho=0.3, print_trace=False):
    n = len(access)
    S = rela[frontend - 1]
    P = [[0 for col in range(n)] for row in range(n)]
    for i in range(n):
        for j in range(n):
            if access[i][j] != 0:
                P[i][j] = abs(S[j])
    P = guiyi(P)
    M = np.zeros([n, n, n])
    # Forward probability
    for i in range(n):
        for j in range(n):
            if access[i][j] > 0:
                for k in range(n):
                    M[k, i, j] = (1 - beta) * P[k][i] + beta * P[i][j]
    # Normalize w.r.t. out nodes
    for k in range(n):
        for i in range(n):
            if np.sum(M[k, i]) > 0:
                M[k, i] = M[k, i] / np.sum(M[k, i])
    # Add backward edges
    for k in range(n):
        for i in range(n):
            in_inds = []
            for j in range(n):
                if access[i][j] == 0 and access[j][i] != 0:
                    M[k, i, j] = rho * ((1 - beta) * P[k][i] + beta * P[j][i])
                    in_inds.append(j)
            # Normalize wrt in nodes
            if np.sum(M[k, i, in_inds]) > 0:
                M[k, i, in_inds] /= np.sum(M[k, i, in_inds])
    # Add self edges
    for k in range(n):
        for i in range(n):
            if M[k, i, i] == 0:
                in_out_node = list(range(n))
                in_out_node.remove(i)
                M[k, i, i] = max(0, S[i] - max(M[k, i, in_out_node]))
    # Normalize all
    for k in range(n):
        for i in range(n):
            if np.sum(M[k, i]) > 0:
                M[k, i] /= np.sum(M[k, i])

    label = [i for i in range(1, n + 1)]
    # l = monitorrange(road, rankPaces, fronted, label)  # relaToRank = 16
    l = secondorder_randomwalk(M, rankPaces, frontend, label, print_trace=print_trace)
    # print(l)
    return l, P, M


def test_cloud_ranger(
    data_source="real_micro_service",
    pc_aggregate=5,
    pc_alpha=0.1,
    testrun_round=1,
    frontend=18,
    true_root_cause=[6, 13, 28, 30, 31],
    beta=0.3,
    rho=0.2,
    save_data_fig=False,
    verbose=False,
    runtime_debug=False,
    *args,
    **kws
):
    """
    Params:
        save_data_fig: whether save transition matrix and the graph
        runtime_debug: whether enable runtime debug mode, where each process is always executed.
    """
    np.random.seed(42)
    random.seed(42)

    if verbose:
        # verbose level >= 1: print method name
        print("{:#^80}".format("Cloud Ranger"))
        if verbose>=2:
            # verbose level >= 2: print method parameters
            print("{:-^80}".format(data_source))
            print("{:^10}pc_aggregate  :{}".format("", pc_aggregate))
            print("{:^10}pc_alpha      :{}".format("", pc_alpha))
            print("{:^10}beta          :{}".format("", beta))
            print("{:^10}rho           :{}".format("", rho))
    # region Load and preprocess data
    # if raw_data not provided in kws
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
        
    # endregion

    rela = calc_pearson(data, method="numpy", zero_diag=False)

    # The file name for saving dependency graph
    window_start=0
    if 'window_start' in kws:
        window_start=kws['window_start']
    dep_graph_filepath = os.path.join(
        "netmedic",
        "results",
        data_source,
        "dep_graph_agg{}_alpha{}_winstart{}_len{}.xlsx".format(pc_aggregate, pc_alpha, window_start, data.shape[1]),
    )

    # When PC dep_graph isn't given, use PC algorithm
    if 'dep_graph' not in kws:
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
    access = dep_graph

    topk_list = range(1, 6)
    prkS = [0] * len(topk_list)
    acc = 0
    for i in range(testrun_round):
        if verbose and verbose >= 2:
            # verbose level >= 2: print random walk starting info
            print("{:^15}Randwalk round:{}".format("", i))
            print(
                "{:^15}Starting randwalk at({}): {}".format(
                    "", frontend, data_head[frontend - 1]
                )
            )
        rank, P, M = relaToRank(
            rela, access, 10, frontend, beta=beta, rho=rho, print_trace=False
        )
        for j, k in enumerate(topk_list):
            prkS[j] += prCal(rank, k, true_root_cause)
        acc += my_acc(rank, true_root_cause, n=len(data_head))
        if verbose and verbose >= 2:
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
        for i in range(len(data_head)):
            saveToExcel(
                os.path.join(
                    "cloud_ranger",
                    "results",
                    data_source,
                    "transition_prob_agg{}_alpha{}_prev{}.xlsx".format(
                        pc_aggregate, pc_alpha, i + 1
                    ),
                ),
                M[i].tolist(),
            )
            draw_weighted_graph(
                M[i].tolist(),
                os.path.join(
                    "cloud_ranger",
                    "results",
                    data_source,
                    "transition_graph_agg{}_alpha{}_prev{}.png".format(
                        pc_aggregate, pc_alpha, i + 1
                    ),
                ),
            )
        draw_weighted_graph(
            access,
            os.path.join(
                "cloud_ranger",
                "results",
                data_source,
                "access_agg{}_alpha{}.png".format(pc_aggregate, pc_alpha),
            ),
        )
    return prkS, acc


if __name__ == '__main__':
    # real_micro_service test suite
    print("\n{:!^80}\n".format(" Real Micro Service Test Suite "))

    entry_point_list = [14]
    true_root_cause = [6, 28, 30, 31]
    verbose=False
    tic = time.time()
    test_cloud_ranger(
        data_source="real_micro_service",
        pc_aggregate=5,
        pc_alpha=0.1,
        testrun_round=5,
        beta=0.1,
        rho=0.2,
        frontend=entry_point_list[0],
        true_root_cause=true_root_cause,
        verbose=verbose,
    )
    toc = time.time() - tic
    print('Used time: {} s'.format(toc))