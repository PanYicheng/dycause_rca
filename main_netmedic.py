import os

import numpy as np

from util_funcs.draw_graph import draw_weighted_graph
from util_funcs.evaluation_function import pr_stat, my_acc, print_prk_acc
from util_funcs.excel_utils import readExl, saveToExcel
from util_funcs.pearson import calc_pearson
from util_funcs.loaddata import load, aggregate
from cloud_ranger.get_link_matrix import build_graph_pc
from netmedic.compute_functions import *


def test_netmedic(
    data_source="real_micro_service",
    history_range=(0, -100),
    current_range=(-100, None),
    bin_size=100,
    affected_node=14,
    true_root_cause=[28],
    verbose=False,
    runtime_debug=False,
    pc_aggregate=1,
    pc_alpha=0.1,
    *args,
    **kws
):
    """
    Params:
        runtime_debug: whether enable runtime debug mode, where each process is always executed.
    """
        
    if verbose:
        # verbose level >= 1: print method name
        print("{:#^80}".format(" Net Medic "))
        if verbose>=2:
            # verbose level >= 2: print method parameters
            print("{:-^80}".format(data_source))
            print("{:^10}history range  :".format(""), history_range)
            print("{:^10}current range  :".format(""), current_range)
            print("{:^10}bin_size       :".format(""), bin_size)

    path_output = "netmedic/results/" + data_source
    # Load all data and appropriately preprocess
    # if data not provided in kws
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
        pc_data = data
    else:
        data_head = kws['data_head']
        # raw_data is of shape [T, N]. Here we first transpose it to [N, T] and then aggregate
        # to get pc_data. NetMedic uses raw data, so we just transpose it ot shape [N, T].
        raw_data = kws['data']
        pc_data = np.array([aggregate(row, pc_aggregate) for row in raw_data.T])
        data = raw_data.T

    # The file name for saving dependency graph
    window_start=0
    if 'window_start' in kws:
        window_start=kws['window_start']
    dep_graph_filepath = os.path.join(
        "netmedic",
        "results",
        data_source,
        "dep_graph_agg{}_alpha{}_winstart{}_len{}.xlsx".format(pc_aggregate, pc_alpha, window_start, pc_data.shape[1]),
    )

    # When PC dep_graph isn't given, use PC algorithm
    if 'dep_graph' not in kws:
        # Build dependency graph either from ground truth or by PC
        if data_source == "pymicro":
            dep_graph = readExl(os.path.join("data", data_source, "true_access.xlsx"))
        elif data_source == "real_micro_service":
            # If it is not in runtime_debug mode and cached file exists, 
            # load previous constructed graph
            if os.path.exists(dep_graph_filepath) and not runtime_debug:
                # If previous dependency graph exists, load it.
                if verbose and verbose >= 2:
                    # verbose level >= 2: print dependency graph loading info
                    print(
                        "{:^10}Loading existing depgraph file: {}".format(
                            "", dep_graph_filepath
                        )
                    )
                dep_graph = readExl(dep_graph_filepath)
            else:
                # If previous dependency graph doesn't exist, genereate it using PC algorithm.
                if verbose and verbose >= 2:
                    # verbose level >= 2: print dependency graph construction info
                    print("{:^10}Generating new depgraph".format(""))
                dep_graph = build_graph_pc(pc_data, alpha=pc_alpha)
    # When PC dep_graph is given, use dep_graph given
    else:
        dep_graph = kws['dep_graph']
    # If not in runtime debugging mode, cache dependency graph
    if not runtime_debug:
        os.makedirs(os.path.dirname(dep_graph_filepath), exist_ok=True)
        saveToExcel(dep_graph_filepath, dep_graph)
    impact_graph = np.array(dep_graph)

    # Compute abnormality
    abnormality = compute_abnormality(data, history_range, current_range)
    if verbose and verbose >= 2:
        # verbose level >= 2: print service abnormality info
        for i in range(data.shape[0]):
            print("Abnormality of service {}: {}".format(i + 1, abnormality[i]))
    

    # Compute Edge weights
    edge_weight = compute_edgeweight(
        data, impact_graph, history_range, current_range, bin_size
    )
    

    # Ranking root causes
    impact_matrix = compute_impact_matrix(impact_graph, edge_weight)
    saveToExcel(os.path.join(path_output, "impact_matrix.xlsx"), edge_weight.tolist())
    global_impact = compute_global_impact(impact_matrix, abnormality)
    # If it's not runtime_debug mode, save all intermidiate results
    if not runtime_debug:
        # Service abnormality scores
        saveToExcel(
                os.path.join(path_output, "abnormality.xlsx"),
                abnormality.reshape(-1, 1).tolist(),
        )
        # Edge weight between services
        saveToExcel(os.path.join(path_output, "edge_weight.xlsx"), edge_weight.tolist())
        # Global abnormality impact of each service
        saveToExcel(
            os.path.join(path_output, "global_impact.xlsx"),
            global_impact.reshape(-1, 1).tolist(),
        )
    node_score = []
    for i in range(data.shape[0]):
        node_score.append(
            (i + 1, 1 / (impact_matrix[i, affected_node - 1] * global_impact[i]))
        )
    node_score.sort(key=lambda x: x[1])

    # Evaluate and print info
    if verbose and verbose >= 2:
        # verbose level >= 2: print service ranking
        print("{:^10}\n{:^10}".format("Ranked nodes:", ""), node_score)
    prkS = pr_stat(node_score, true_root_cause)
    acc = my_acc(node_score, true_root_cause, n=len(data_head))
    # Display PR@k and Acc if disable_print is not set
    if 'disable_print' not in kws or kws['disable_print'] is False:
        print_prk_acc(prkS, acc)
    return prkS, acc
