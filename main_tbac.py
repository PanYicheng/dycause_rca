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


def get_callers(access, node, search_depth=1):
    """Find the callers of node, using fixed search depth
    """
    callers = []
    queue = [node]
    visited = [node]
    current_depth = 1
    while len(queue) > 0 and search_depth > 0:
        front = queue.pop()
        for j in range(len(access)):
            if access[front][j] > 0 and j not in visited:
                queue.append(j)
                callers.append([j, current_depth])
                visited.append(j)
        search_depth -= 1
        current_depth += 1
    return callers


def get_callees(access, node, search_depth=1):
    """Find the callers of node, using fixed search depth
    """
    callees = []
    queue = [node]
    visited = [node]
    current_depth = 1
    while len(queue) > 0 and search_depth > 0:
        front = queue.pop()
        for j in range(len(access)):
            if access[j][front] > 0 and j not in visited:
                queue.append(j)
                callees.append([j, current_depth])
                visited.append(j)
        search_depth -= 1
        current_depth += 1
    return callees


def weighted_powermean(score, p, weight):
    """Weighted power mean in TBAC

    Params:
        score: scores list
        p: power exponent
        weight: weight for each score, same length as s
    """

    def f(a, p):
        if a >= 0:
            return np.power(a, p)
        else:
            return np.power(-a, p) * (-1.0)

    s = 0.0
    for i in range(len(score)):
        s += weight[i] * f(score[i], p)
    s /= sum(weight)
    return f(s, 1.0 / p)


def correlation_algorithm(scores, access):
    def check_greater(a, b):
        if a is None or b is None:
            return True
        else:
            return a > b

    def check_lessequal(a, b):
        if a is None or b is None:
            return True
        else:
            return a <= b

    n = len(access)
    rating = scores.copy()
    for node in range(n):
        callers = get_callers(access, node, search_depth=20)
        callees = get_callees(access, node, search_depth=20)
        S_in_mean = (
            weighted_powermean(
                [scores[_[0]] for _ in callers], 1, [1 / _[1] for _ in callers]
            )
            if len(callers) > 0
            else None
        )
        S_out_max = (
            np.max(scores[[_[0] for _ in callees]]) if len(callees) > 0 else None
        )
        if check_greater(S_in_mean, scores[node]) and check_lessequal(
            S_out_max, scores[node]
        ):
            rating[node] = 0.5 * (scores[node] + 1)
        if check_lessequal(S_in_mean, scores[node]) and check_greater(
            S_out_max, scores[node]
        ):
            rating[node] = 0.5 * (scores[node] - 1)
    return rating


def test_tbac(
    data_source="pymicro", 
    pc_aggregate=5, 
    pc_alpha=0.1, 
    frontend=16, 
    true_root_cause=[1],
    verbose=False,
    runtime_debug=False,
    *args,
    **kws
):
    np.random.seed(42)
    random.seed(42)

    if verbose:
        # verbose level >= 1: print method name
        print("{:#^80}".format("TBAC"))
        if verbose>1:
            # verbose level >= 2: print data load parameters
            print("{:-^80}".format(data_source))
            print("{:^10}pc_aggregate :{}".format("", pc_aggregate))
            print("{:^10}frontend     :{}".format("", frontend))
    
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
    if verbose and verbose >= 2:
        # verbose level >= 2: print data load parameters
        print("{:^10}Data shape:".format(""), data.shape)
    
    rela = calc_pearson(data, method="numpy", zero_diag=False)
    
    # The file name for caching dependency graph
    window_start=0
    if 'window_start' in kws:
        window_start=kws['window_start']
    access_filepath = os.path.join(
            "tbac",
            "results",
            data_source,
            "access_agg{}_alpha{}_winstart{}_len{}.xlsx".format(pc_aggregate, pc_alpha, window_start,data.shape[1]),
    )
    
    # When PC dep_graph isn't given, use PC algorithm
    if 'dep_graph' not in kws:
        if data_source == "pymicro":
            # Real call topology matrix
            access = readExl(os.path.join("data", data_source, "true_access.xlsx"))
        elif data_source == "real_micro_service":
            # If it is not in runtime_debug mode and cached file exists, load previous constructed graph
            if os.path.exists(access_filepath) and not runtime_debug:
                # If previous dependency graph exists, load it.
                if verbose and verbose > 1:
                    # verbose level >= 2: print dependency graph construction info
                    print(
                        "{:^10}Loading existing link matrix file: {}".format(
                            "", access_filepath
                        )
                    )
                access = readExl(access_filepath)
            else:
                # If cached dependency graph doesn't exist, genereate it using PC algorithm.
                if verbose and verbose > 1:
                    # verbose level >= 2: print dependency graph construction info
                    print("{:^10}Generating new link matrix".format(""))
                # Use PC algorithm to create access matrix
                access = build_graph_pc(data, alpha=pc_alpha)
    # When PC dep_graph is given, use dep_graph given
    else:
        access = kws['dep_graph']
    # If not in runtime debugging mode, cache dependency graph
    if not runtime_debug:
        os.makedirs(os.path.dirname(access_filepath), exist_ok=True)
        saveToExcel(access_filepath, access)

    # Caculate node anomaly score
    #   using correlation to frontend node, here
    #   translate its range to [-1, 1]
    anomaly_score = np.array(rela[frontend - 1]) * 2 - 1

    # Correlation algorithm in TBAC
    rating = correlation_algorithm(anomaly_score, access)

    rank = list(zip(range(1, len(data) + 1), rating.tolist()))
    rank.sort(key=lambda x: x[1], reverse=True)
    if verbose and verbose >= 2:
        # verbose level >= 2: print root cause ranks
        print("{:^15}".format(""), end="")
        for j in range(5):
            print(rank[j], end=", ")
        print("")

    topk_list = range(1, 6)
    prkS = [0] * len(topk_list)
    for k in range(1, 6):
        prkS[k - 1] += prCal(rank, k, true_root_cause)
    acc = my_acc(rank, true_root_cause)
    # Display PR@k and Acc if disable_print is not set
    if 'disable_print' not in kws or kws['disable_print'] is False:
        print_prk_acc(prkS, acc)
    return prkS, acc
