"""Test and compare all algorithms under different input windows. 
Here, we change the window length with a fixed center period, which contains
abnormal data. 
The windows overlap at the abnormal data range, e.g. [4653, 4853].
"""
import os
import sys
import time
import pickle

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import Manager

from main_cloud_ranger import test_cloud_ranger
from main_dycause import test_dycause
from main_tbac import test_tbac
from main_monitor_rank import test_monitor_rank
# from main_facgraph import test_facgraph
from main_netmedic import test_netmedic
from cloud_ranger.get_link_matrix import build_graph_pc
from util_funcs.loaddata import aggregate
from util_funcs.excel_utils import readExl, saveToExcel


def load_data(data_source):
    """Load data

    Params:
        data_source: data location, can be 'pymicro' or 'ibm_micro_service'

    Returns:
        data       : numpy array of the data, each column is a variable, shape [T, N]    
    """
    from utils.loaddata import load
    data, data_head = load(
        os.path.join("data", data_source, "rawdata.xlsx"),
        normalize=True,
        zero_fill_method='prevlatter',
        aggre_delta=1,
        verbose=False,
    )
    return data, data_head


def build_common_pc_graph(data, aggre_delta, alpha, window_start):
    """Build the PC dependency graph for all needed methods.
    Execution time is recorded and returned. 
    May cache the calculated result and reuse later.

    Params:
        data       : input data in numpy array, shape [T, N]
        aggre_delta: aggregation size, change the data resolution
        alpha      : siginificance value in PC algorithm
        window_start: the window start position in raw data

    Returns: 
        dep_graph: generated dependency graph, in matrix format
        toc      : execution time of PC algorithm, -1 means that using cached file
    """
    cache_filename = os.path.join('tmp', 'pc_dep_graph_start{}_len{}_agg{}_alpha{}.xlsx'.format(window_start, data.shape[0],
                                                                                                aggre_delta, alpha))
    os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
    if os.path.exists(cache_filename):
        print("{:^10}Loading previous PC dependency graph".format(''))
        dep_graph = readExl(cache_filename)
        toc = -1
    else:
        print("{:^10}Starting PC dependency graph construction".format(''))
        tic = time.time()
        # Transpose to aggregate
        pc_data = [aggregate(row, aggre_delta) for row in data.T]
        dep_graph = build_graph_pc(pc_data, alpha=0.1)
        toc = time.time() - tic
        print("{:^10}PC dependency graph construction finished! Used: {} seconds".format(
            '', toc))
        saveToExcel(cache_filename, dep_graph)
    return dep_graph, toc


def tbac_param_search(pc_aggregate, pc_alpha, entry_point, true_root_cause, data_inwindow, data_head, dep_graph,
                      window_start):
    """TBAC parameter searching function. Iterates over possible parameters except PC parameters.
    """
    tic = time.time()
    prks, acc = test_tbac(
        "ibm_micro_service",
        pc_aggregate=pc_aggregate,
        pc_alpha=pc_alpha,
        frontend=entry_point,
        true_root_cause=true_root_cause,
        verbose=0,
        runtime_debug=False,
        data=data_inwindow,
        data_head=data_head,
        disable_print=True,
        dep_graph=dep_graph,
        window_start=window_start
    )
    toc = time.time()-tic
    return [{
        'time': toc,
        'prks': prks,
        'acc': acc}]


def netmedic_param_search(entry_point, true_root_cause,
                          pc_aggregate, pc_alpha, data_inwindow, data_head, dep_graph, window_start):
    """NetMedic parameter searching function. Iterates over possible parameters except PC parameters.
    """
    result_list = []
    for hist_start in range(0, 200, 50):
        for hist_len in [50]:
            for current_start in range(200, 300, 50):
                for current_len in[50]:
                    for bin_size in [5, 10, 15]:
                        tic = time.time()
                        prks, acc = test_netmedic(
                            data_source="ibm_micro_service",
                            history_range=(hist_start, hist_start+hist_len),
                            current_range=(
                                current_start, current_start+current_len),
                            bin_size=bin_size,
                            affected_node=entry_point,
                            true_root_cause=true_root_cause,
                            verbose=0,
                            runtime_debug=False,
                            pc_aggregate=pc_aggregate,
                            pc_alpha=pc_alpha,
                            data=data_inwindow,
                            data_head=data_head,
                            disable_print=True,
                            dep_graph=dep_graph,
                            window_start=window_start
                        )
                        toc = time.time()-tic
                        result_list.append({
                            'history_range': (hist_start, hist_start+hist_len),
                            'current_range': (current_start, current_start+current_len),
                            'bin_size': bin_size,
                            'time': toc,
                            'prks': prks,
                            'acc': acc
                        })
    return result_list


def monitorrank_param_search(pc_aggregate, pc_alpha, entry_point, true_root_cause, data_inwindow,
                             data_head, dep_graph, window_start):
    """Monitor Rank parameter searching function. Iterates over possible parameters except PC parameters.
    """
    result_list = []
    for rho in np.arange(0.1, 1.0, 0.2):
        tic = time.time()
        prks, acc = test_monitor_rank(
            "ibm_micro_service",
            pc_aggregate=pc_aggregate,
            pc_alpha=pc_alpha,
            testrun_round=5,
            rho=rho,
            frontend=entry_point,
            true_root_cause=true_root_cause,
            save_data_fig=False,
            verbose=0,
            runtime_debug=False,
            disable_print=True,
            data=data_inwindow,
            data_head=data_head,
            dep_graph=dep_graph,
            window_start=window_start
        )
        toc = time.time()-tic
        result_list.append({
            'rho': rho,
            'time': toc,
            'prks': prks,
            'acc': acc
        })
        print("Monitor Rank rho:{:.2f} time:{:.4f} acc:{:.4f}".format(rho, toc, acc))
    return result_list


def cloudranger_param_search(ela, alpha, entry_point, true_root_cause, data_inwindow, data_head,
                             dep_graph, window_start):
    """CloudRanger parameter searching function. Iterates over possible parameters except PC parameters.
    """
    result_list = []
    for beta in np.arange(0.1, 1.0, 0.2):
        for rho in np.arange(0.1, 1.0, 0.2):
            tic = time.time()
            prks, acc = test_cloud_ranger(
                data_source="ibm_micro_service",
                pc_aggregate=ela,
                pc_alpha=alpha,
                testrun_round=5,
                beta=beta,
                rho=rho,
                frontend=entry_point,
                true_root_cause=true_root_cause,
                verbose=0,
                runtime_debug=False,
                data=data_inwindow,
                data_head=data_head,
                disable_print=True,
                dep_graph=dep_graph,
                window_start=window_start
            )
            toc = time.time()-tic
            result_list.append({
                'ela': ela,
                'beta': beta,
                'rho': rho,
                'time': toc,
                'prks': prks,
                'acc': acc
            })
            print("Cloudranger ela:{:d} beta:{:.2f} rho:{:.2f} time:{:.4f} acc:{:.4f}".format(
                ela, beta, rho, toc, acc))
    return result_list


def dycause_param_search(entry_point, true_root_cause, data_inwindow, data_head, window_start):
    """Granger extend parameter searching function. Iterates over possible parameters except PC parameters.
    """
    result_list = []
    runtime_debug = True
    for pre_length in [0]:
        for post_length in [200]:
            # Skip experiments that takes less than 200 seconds data
            if pre_length + post_length < 200:
                continue
            for step in [50]:
                for lag in [15]:
                    for thres in [0.5]:
                        tic = time.time()
                        prks, acc = test_dycause(
                            # Data params
                            data_source="ibm_micro_service",
                            aggre_delta=1,
                            start_time=None,
                            before_length=pre_length,
                            after_length=post_length,
                            # Granger interval based graph construction params
                            step=step,
                            significant_thres=0.1,
                            lag=lag,  # must satisfy: step > 3 * lag + 1
                            auto_threshold_ratio=thres,
                            # Root cause analysis params
                            testrun_round=1,
                            frontend=entry_point,
                            true_root_cause=true_root_cause,
                            max_path_length=None,
                            mean_method="harmonic",
                            topk_path=150,
                            num_sel_node=3,
                            # Debug params
                            plot_figures=False,
                            verbose=0,
                            runtime_debug=runtime_debug,
                            data=data_inwindow,
                            data_head=data_head,
                            disable_print=True,
                            window_start=window_start
                        )
                        toc = time.time() - tic
                        result_list.append({
                            'pre_len': pre_length,
                            'post_len': post_length,
                            'step': step,
                            'lag': lag,
                            'auto_threshold_ratio': thres,
                            'testrun_round': 1,
                            'runtime_debug': runtime_debug,
                            'time': toc,
                            'prks': prks,
                            'acc': acc
                        })
                        print("Granger extend pre:{:d} post:{:d} step:{:d} "
                            "lag:{:d} thres:{:.2f} time:{:.4f} acc:{:.4f}".format(
                            pre_length, post_length, step, lag, thres, toc, acc))
    return result_list


if __name__ == '__main__':
    # ibm_micro_service test suite
    print("\n{:!^80}\n".format(" IBM Micro Service Test Suite "))

    entry_point_list = [14]
    true_root_cause = [6, 28, 30, 31]
    verbose = 0
    disable_print = True

    result_dict = {}
    # Append one result for method to result_dict

    def append_method_result(result_dict, method_name, result):
        if method_name in result_dict:
            result_dict[method_name].append(result)
        else:
            result_dict[method_name] = [result]
    # Load all range data
    data, data_head = load_data("ibm_micro_service")
    # window_len = 1200
    pbar = tqdm(total=3, ascii=True)
    # Center 4753
    center_index = 4753
    for window_len in range(200, 7200, 1200):
    # for window_len in [3600]:
        window_start = int(center_index - window_len/2)
        print("{:<40}Window start: {:d}, len: {:d}".format("", window_start, window_len))
        # Slice the data in range [window_start: window_start+300]
        data_inwindow = data[
            max(0, min(window_start, data.shape[0]-1)):\
            max(0, min(window_start+window_len, data.shape[0]-1)), :]
        # if len(data_inwindow) < 200:
        #     # If doesn't contain enough data samples
        #     break

        for pc_aggregate in [20]:
            for pc_alpha in [0.1]:
                dep_graph, pc_time = build_common_pc_graph(data_inwindow, pc_aggregate, pc_alpha,
                                                            window_start)
                append_method_result(result_dict, 'pc algorithm', {
                    'window_start': window_start,
                    'window_len': window_len,
                    'pc_aggregate': pc_aggregate,
                    'pc_alpha': pc_alpha,
                    'pc_time': pc_time,
                })

                result_list = tbac_param_search(pc_aggregate, pc_alpha, entry_point_list[0], true_root_cause, data_inwindow, data_head, dep_graph,
                                                window_start)
                for result in result_list:
                    result['window_start'] = window_start
                    result['window_len'] = window_len
                    result['pc_aggregate'] = pc_aggregate
                    result['pc_alpha'] = pc_alpha
                    append_method_result(result_dict, 'tbac', result)

                result_list = netmedic_param_search(entry_point_list[0], true_root_cause,
                                                    pc_aggregate, pc_alpha, data_inwindow, data_head, dep_graph, 
                                                    window_start)
                for result in result_list:
                    result['window_start'] = window_start
                    result['window_len'] = window_len
                    result['pc_aggregate'] = pc_aggregate
                    result['pc_alpha'] = pc_alpha
                    append_method_result(result_dict, 'netmedic', result)

                result_list = monitorrank_param_search(pc_aggregate, pc_alpha, entry_point_list[0],
                                                        true_root_cause, data_inwindow, data_head, dep_graph,
                                                        window_start)
                for result in result_list:
                    result['window_start'] = window_start
                    result['window_len'] = window_len
                    result['pc_aggregate'] = pc_aggregate
                    result['pc_alpha'] = pc_alpha
                    append_method_result(result_dict, 'monitorrank', result)

                result_list = cloudranger_param_search(pc_aggregate, pc_alpha, entry_point_list[0],
                                                        true_root_cause, data_inwindow, data_head,  dep_graph,
                                                        window_start)
                for result in result_list:
                    result['window_start'] = window_start
                    result['window_len'] = window_len
                    result['pc_aggregate'] = pc_aggregate
                    result['pc_alpha'] = pc_alpha
                    append_method_result(result_dict, 'cloudranger', result)

        # granger extend test
        result_list = dycause_param_search(
            entry_point_list[0], true_root_cause, data_inwindow, data_head, window_start)
        for result in result_list:
            result['window_start'] = window_start
            result['window_len'] = window_len
            append_method_result(result_dict, 'grangerextend', result)
        pbar.update(1)
        # break

    with open('test_all_perf_ibm_overlappedwindows.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
    pbar.close()
