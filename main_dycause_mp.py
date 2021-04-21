import argparse
from collections import defaultdict
import datetime
import threading
import os
import sys
import time
import pickle
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import Manager

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm

from dycause_lib.anomaly_detect import anomaly_detect
# loop_granger是Granger causal interval 作者提供的代码
from dycause_lib.Granger_all_code import loop_granger
from dycause_lib.causal_graph_build import get_segment_split
from dycause_lib.causal_graph_build import get_ordered_intervals
from dycause_lib.causal_graph_build import get_overlay_count
from dycause_lib.causal_graph_build import normalize_by_row, normalize_by_column
from dycause_lib.randwalk import randwalk
from dycause_lib.ranknode import ranknode, analyze_root
from dycause_lib.draw_graph import *
from util_funcs.loaddata import load
from util_funcs.draw_graph import draw_weighted_graph
from util_funcs.evaluation_function import prCal, my_acc, pr_stat, print_prk_acc
from util_funcs.format_ouput import format_to_excel
from util_funcs.excel_utils import saveToExcel


def granger_process(
        shared_params_dict,
        specific_params,
        shared_result_dict):
    try:
        # with open(common_params_filename, 'rb') as f:
        #     common_params = pickle.load(f)
        common_params = shared_params_dict
        ret = loop_granger(
            common_params['local_data'],
            common_params['data_head'],
            common_params['dir_output'],
            common_params['data_head'][specific_params['x_i']],
            common_params['data_head'][specific_params['y_i']],
            common_params['significant_thres'],
            common_params['method'],
            common_params['trip'],
            common_params['lag'],
            common_params['step'],
            common_params['simu_real'],
            common_params['max_segment_len'],
            common_params['min_segment_len'],
            verbose=False,
            return_result=True,
        )
    except Exception as e:
        print("Exception occurred at {} -> {}!".format(
            specific_params['x_i'], specific_params['y_i']), e)
        logging.error("Exception occurred at {} -> {}!".format(
            specific_params['x_i'], specific_params['y_i']))
        ret = (None, None, None, None, None)
    shared_result_dict['{}->{}'.format(specific_params['x_i'], specific_params['y_i'])] = ret
    return ret


def test_dycause(
    # Data params
    data_source="real_micro_service",
    aggre_delta=1,
    start_time=None,
    before_length=300,
    after_length=300,
    # Granger interval based graph construction params
    step=50,
    significant_thres=0.05,
    lag=5,  # must satisfy: step > 3 * lag + 1
    auto_threshold_ratio=0.8,
    runtime_debug=False,
    # Root cause analysis params
    testrun_round=1,
    frontend=14,
    max_path_length=None,
    mean_method="arithmetic",
    true_root_cause=[28],
    topk_path=60,
    num_sel_node=1,
    # Debug params
    plot_figures=False,
    verbose=True,
    max_workers=5,
    **kws,
):
    """
    Params:
        plot_figures: whether plot result figures. Can be a list of figure names, such as ['all-data', 'abnormal-data', 
            'dycurves', 'aggre-imgs', 'graph']. Can also True for enable all figure plots, False for disable all.
        runtime_debug: whether enable runtime debug mode, where loop_granger is always executed.
    """
    if runtime_debug:
        time_stat_dict = {}
        tic = time.time()
    if 'disable_print' not in kws or kws['disable_print'] is False:
        print("{:#^80}".format(" DyCause "))

    dir_output = "dycause/results/" + data_source
    os.makedirs(dir_output, exist_ok=True)
    if verbose:
        print("{:-^80}".format("Data load phase"))
    # region Load and preprocess data
    data, data_head = load(
        os.path.join("data", data_source, "rawdata.xlsx"),
        normalize=True,
        zero_fill_method='prevlatter',
        aggre_delta=aggre_delta,
        verbose=verbose,
    )
    #  Plot all data if asked
    if (plot_figures is True) or (isinstance(plot_figures, list) and 'all-data' in plot_figures):
        draw_alldata(
            data,
            data_head,
            os.path.join(dir_output, "all-data-L{}.png".format(data.shape[0])),
        )
    # endregion

    # region Set start time in data to analyze if not provided
    anomaly_score = 'Not calculated'
    if start_time is None:
        start_time, anomaly_score = anomaly_detect(
            data,
            weight=1,
            mean_interval=50,
            anomaly_proportion=0.3,
            verbose=verbose,
            save_fig=(plot_figures is True) or (isinstance(
                plot_figures, list) and 'anomaly-score' in plot_figures),
            path_output=dir_output,
        )
    if verbose:
        print(
            "{space:^10}{name1:<30}: {}\n"
            "{space:^10}{name2:<30}: {}".format(
                start_time,
                anomaly_score,
                space="",
                name1="Start time",
                name2="Abnormal score",
            )
        )
    # plot abnormal data of each services if asked
    if (plot_figures is True) or (isinstance(plot_figures, list) and 'abnormal-data' in plot_figures):
        draw_alldata(
            data[start_time - before_length: start_time + after_length, :],
            data_head,
            os.path.join(
                dir_output,
                "abnomal-data-plot-S{}-E{}.png".format(
                    start_time - before_length, start_time + after_length
                ),
            ),
        )
    # endregion
    if runtime_debug:
        toc = time.time()
        time_stat_dict['Load phase'] = toc-tic
        tic = toc

    # region Run loop_granger to get the all intervals
    if verbose:
        print("{:-^80}".format("Granger interval based impact graph construction phase"))
    local_length = before_length + after_length
    local_data = data[start_time - before_length: start_time + after_length, :]
    method = "fast_version_3"
    trip = -1
    simu_real = "simu"
    max_segment_len = before_length + after_length
    min_segment_len = step
    list_segment_split = get_segment_split(before_length + after_length, step)

    local_results_file_path = os.path.join(
        dir_output,
        "local-results",
        "aggregate-{}".format(aggre_delta),
        "local_results"
        "_start{start}_bef{bef}_aft{aft}_lag{lag}_sig{sig}_step{step}_min{min}_max{max}.pkl".format(
            start=start_time,
            bef=before_length,
            aft=after_length,
            lag=lag,
            sig=significant_thres,
            step=step,
            min=min_segment_len,
            max=max_segment_len,
        ),
    )
    if os.path.exists(local_results_file_path) and not runtime_debug:
        if verbose:
            print(
                "{:^10}".format(
                    "") + "Loading previous granger interval results:",
                os.path.basename(local_results_file_path),
            )
        with open(local_results_file_path, "rb") as f:
            local_results = pickle.load(f)
    else:
        if verbose:
            print(
                "{space:^10}{name}:\n"
                "{space:^15}bef len      :{bef}\n"
                "{space:^15}aft len      :{aft}\n"
                "{space:^15}lag          :{lag}\n"
                "{space:^15}significant  :{sig}\n"
                "{space:^15}step         :{step}\n"
                "{space:^15}min len      :{min}\n"
                "{space:^15}max len      :{max}\n"
                "{space:^15}segment split:".format(
                    space="",
                    name="Calculating granger intervals",
                    bef=before_length,
                    aft=after_length,
                    lag=lag,
                    sig=significant_thres,
                    step=step,
                    min=min_segment_len,
                    max=max_segment_len,
                ),
                list_segment_split,
            )
        local_results = defaultdict(dict)

        # region normal single thread version
        # for x_i in range(len(data_head)):
        #     for y_i in range(len(data_head)):
        #         if x_i == y_i:
        #             continue
        #         feature = data_head[x_i]
        #         target = data_head[y_i]
        #         (total_time, time_granger, time_adf, array_results_YX,
        #          array_results_XY) = granger_process(x_i, y_i)
        #         print('Iter {:2d}->{:2d} '
        #               'Total time  :{:5.4f} '
        #               'Granger time:{:5.4f} '
        #               'Adf time    :{:5.4f}'.format(x_i, y_i,
        #                                             total_time,
        #                                             time_granger,
        #                                             time_adf),
        #               end='\r')
        #         matrics = [array_results_YX, array_results_XY]
        #         ordered_intervals = get_ordered_intervals(
        #             matrics, significant_thres, list_segment_split)
        #         local_results['%s->%s' %
        #                       (x_i, y_i)]['intervals'] = ordered_intervals
        #         local_results['%s->%s' %
        #                       (x_i, y_i)]['result_YX'] = array_results_YX
        #         local_results['%s->%s' %
        #                       (x_i, y_i)]['result_XY'] = array_results_XY
        # # skip the \r print line
        # print('')
        # endregion

        # region ThreadPoolExecuter&ProcessPoolExecutor version
        total_thread_num = [len(data_head) * (len(data_head) - 1)]
        thread_results = [0 for i in range(total_thread_num[0])]
        if verbose:
            pbar = tqdm(total=total_thread_num[0], ascii=True)

        common_params_filename = os.path.join(
            dir_output, 'local-results', 'common_params.pkl')
        common_params = {
                'local_data': local_data,
                'data_head': data_head,
                'dir_output': dir_output,
                'significant_thres': significant_thres,
                'method': method,
                'trip': trip,
                'lag': lag,
                'step': step,
                'simu_real': simu_real,
                'max_segment_len': max_segment_len,
                'min_segment_len': min_segment_len
            }
        manager = Manager()
        shared_params_dict = manager.dict()
        shared_result_dict = manager.dict()
        for key, value in common_params.items():
            shared_params_dict[key]=value
        # with open(common_params_filename, 'wb') as f:
        #     pickle.dump(common_params, f)
        executor = ProcessPoolExecutor(max_workers=max_workers)
        i = 0
        futures = []
        tic = time.time()
        for x_i in range(len(data_head)):
            for y_i in range(len(data_head)):
                if x_i == y_i:
                    continue
                futures.append(executor.submit(
                    granger_process,
                    shared_params_dict,
                    {'x_i': x_i, 'y_i': y_i},
                    shared_result_dict
                )
                )
                i = i + 1

        future_complete_time = []
        if verbose:
            for future in as_completed(futures):
                pbar.update(1)
                future_complete_time.append(time.time()-tic)
            pbar.close()
            # save_path = os.path.join(dir_output, 'local-results', 'future-complete-time.pkl')
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # with open(save_path, 'wb') as f:
            #     pickle.dump(future_complete_time, f)
        executor.shutdown(wait=True)
        # print('shared_result_dict keys: ', list(shared_result_dict.keys()))
        # exit(0)
        i = 0
        for x_i in range(len(data_head)):
            for y_i in range(len(data_head)):
                if x_i == y_i:
                    continue
                # (
                #     total_time,
                #     time_granger,
                #     time_adf,
                #     array_results_YX,
                #     array_results_XY,
                # ) = futures[i].result()
                (
                    total_time,
                    time_granger,
                    time_adf,
                    array_results_YX,
                    array_results_XY,
                ) = shared_result_dict['{}->{}'.format(x_i, y_i)]
                
                matrics = [array_results_YX, array_results_XY]
                ordered_intervals = get_ordered_intervals(
                    matrics, significant_thres, list_segment_split
                )
                local_results["%s->%s" %
                              (x_i, y_i)]["intervals"] = ordered_intervals
                local_results["%s->%s" %
                              (x_i, y_i)]["result_YX"] = array_results_YX
                local_results["%s->%s" %
                              (x_i, y_i)]["result_XY"] = array_results_XY
                i = i + 1
        # endregion
        if not runtime_debug:
            # Only save local results if not in runtime debug mode
            os.makedirs(os.path.dirname(
                local_results_file_path), exist_ok=True)
            with open(local_results_file_path, "wb") as f:
                pickle.dump(local_results, f)
    # endregion
    if runtime_debug:
        toc = time.time()
        time_stat_dict['granger causal intervals'] = toc - tic
        tic = toc

    # region Construction impact graph using generated intervals
    # Generate dynamic causal curve between two services by overlaying intervals
    histogram_sum = defaultdict(int)
    edge = []
    edge_weight = dict()
    for x_i in range(len(data_head)):
        for y_i in range(len(data_head)):
            if y_i == x_i:
                continue
            key = "{0}->{1}".format(x_i, y_i)
            intervals = local_results[key]["intervals"]
            overlay_counts = get_overlay_count(local_length, intervals)
            # whether plot temporaray figure pair wise
            if (plot_figures is True) or (isinstance(plot_figures, list) and 'dycurves' in plot_figures):
                os.makedirs(os.path.join(
                    dir_output, "dynamic-causal-curves"), exist_ok=True)
                if verbose:
                    print(
                        "{:^10}Ploting {:2d}->{:2d}".format("", x_i + 1, y_i + 1), end="\r"
                    )
                draw_overlay_histogram(
                    overlay_counts,
                    "{}->{}".format(x_i + 1, y_i + 1),
                    os.path.join(
                        dir_output, "dynamic-causal-curves", "{0}-{1}.png".format(
                            x_i + 1, y_i + 1)
                    ),
                )
            histogram_sum[key] = sum(overlay_counts)
    # skip the \r print line
    if (plot_figures is True) or (isinstance(plot_figures, list) and 'dycurves' in plot_figures) and verbose:
        print("")
    # Make edges from 1 node using comparison and auto-threshold
    for x_i in range(len(data_head[:])):
        bar_data = []
        for y_i in range(len(data_head)):
            key = "{0}->{1}".format(x_i, y_i)
            bar_data.append(histogram_sum[key])
        # whether plot temporary figure from one node
        if (plot_figures is True) or (isinstance(plot_figures, list) and 'aggre-imgs' in plot_figures):
            if not os.path.exists(os.path.join(dir_output, "aggre-imgs")):
                os.makedirs(os.path.join(dir_output, "aggre-imgs"))
            if verbose:
                print("{:^10}Ploting aggre imgs {:2d}".format("", x_i + 1),
                      end="\r")
            draw_bar_histogram(
                bar_data, auto_threshold_ratio,
                "From service {0}".format(x_i + 1),
                os.path.join(dir_output, "aggre-imgs",
                             "{0}.png".format(x_i + 1)),
            )
        bar_data_thres = np.max(bar_data) * auto_threshold_ratio
        for y_i in range(len(data_head)):
            if bar_data[y_i] >= bar_data_thres:
                edge.append((x_i, y_i))
                edge_weight[(x_i, y_i)] = bar_data[y_i]
    # skip the \r print line
    if (plot_figures is True) or (isinstance(plot_figures, list) and 'aggre-imgs' in plot_figures) and verbose:
        print("")
    # Make the transition matrix with edge weight estimation
    transition_matrix = np.zeros([data.shape[1], data.shape[1]])
    for key, val in edge_weight.items():
        x, y = key
        transition_matrix[x, y] = val
    transition_matrix = normalize_by_column(transition_matrix)
    # transition_matrix = normalize_by_row(transition_matrix)

    def save_graph_excel(filename_prefix, matrix):
        common_suffix = "-bef{}-aft{}-step{}-lag{}-thres{}".format(
            before_length,
            after_length,
            step,
            lag, auto_threshold_ratio)
        if (plot_figures is True) or (isinstance(plot_figures, list) and 'graph' in plot_figures):
            draw_weighted_graph(
                matrix,
                os.path.join(
                    dir_output,
                    filename_prefix + common_suffix+".png",
                ),
                weight_multiplier=4,
            )
        saveToExcel(
            os.path.join(
                dir_output,
                filename_prefix + common_suffix+".xlsx",
            ),
            matrix.tolist(),
        )

    save_graph_excel("graph", transition_matrix)
    # endregion
    if runtime_debug:
        toc = time.time()
        time_stat_dict['graph construction'] = toc - tic
        tic = toc

    # region backtrace root cause analysis
    if verbose:
        print("{:-^80}".format("Back trace root cause analysis phase"))
    topk_list = range(1, 6)
    prkS = [0] * len(topk_list)
    if not isinstance(frontend, list):
        frontend = [frontend]
    for entry_point in frontend:
        if verbose:
            print("{:*^40}".format(" Entry: {:2d} ".format(entry_point)))
        prkS_list = []
        acc_list = []
        for i in range(testrun_round):
            ranked_nodes, new_matrix = analyze_root(
                transition_matrix,
                entry_point,
                local_data,
                mean_method=mean_method,
                max_path_length=max_path_length,
                topk_path=topk_path,
                prob_thres=0.2,
                num_sel_node=num_sel_node,
                use_new_matrix=False,
                verbose=verbose,
            )
            if verbose:
                print("{:^0}|{:>8}|{:>12}|".format("", "Node", "Score"))
                for j in range(len(ranked_nodes)):
                    print(
                        "{:^0}|{:>8d}|{:>12.7f}|".format(
                            "", ranked_nodes[j][0], ranked_nodes[j][1]
                        )
                    )
            prkS = pr_stat(ranked_nodes, true_root_cause)
            acc = my_acc(ranked_nodes, true_root_cause, len(data_head))
            prkS_list.append(prkS)
            acc_list.append(acc)
        prkS = np.mean(np.array(prkS_list), axis=0).tolist()
        acc = float(np.mean(np.array(acc_list)))
        if 'disable_print' not in kws or kws['disable_print'] is False:
            print_prk_acc(prkS, acc)

    save_graph_excel("newgraph", new_matrix)
    # endregion
    if runtime_debug:
        toc = time.time()
        time_stat_dict['backtrace rca'] = toc - tic
        tic = toc
        print(time_stat_dict)

    return prkS, acc


parser = argparse.ArgumentParser(description='DyCause root cause analyzer.')
# Create the positional arguments
parser.add_argument('data_source', type=str, default="real_micro_service",
                    help=""""the folder name of the input metric data (the actual file path is """
                        """data/${data_source}/rawdata.xlsx, each row is one service).""")
parser.add_argument('frontend', type=int, help='the entry service of root cause analysis.')
parser.add_argument('root', type=int, nargs="*", help='the root cause services of root cause analysis.')
# Create the optional arguments
parser.add_argument('--agg', default=1, type=int, help='the aggregation delta during preprocessing.')
parser.add_argument('--start', default=None, type=int, help='the anomaly timestamp in input data.')
parser.add_argument('--bef', default=0, type=int, help='the before interval size.')
parser.add_argument('--aft', default=280, type=int, help='the after interval size.')
parser.add_argument('--step', default=70, type=int, help='the minimal step size in Granger causal interval.')
parser.add_argument('--thres', default=0.1, type=float, help='the siginificance threshold in Granger causality test.')
parser.add_argument('--lag', default=5, type=int, help='the maximum causal lag in Granger causality test.')
parser.add_argument('--edge_thres', default=0.5, type=float, help='the edge threshold in dependency graph construction.')
parser.add_argument('--debug', default=False, 
                    help='whether enable runtime debug mode(log calculation time of each step).')
parser.add_argument('--testrun', default=1, type=int, help='how many test rounds.')
parser.add_argument('--max_path', default=None, help='the maximum path length considered in backtrace analysis.')
parser.add_argument('--mean', default="harmonic", help='the mean method used in path probability estimation.')
parser.add_argument('--topk', default=50, type=int, help='the number of top-k paths included in root cause detection.')
parser.add_argument('--num_sel', default=3, type=int, help='the number of considered services in each path as root causes.')
parser.add_argument('--plot', default=None, help="""whether plot result figures. Can be a list of figure names, such as 
                    ['all-data', 'abnormal-data', 'dycurves', 'aggre-imgs', 'graph']. Can also True for enable all 
                    figure plots, False for disable all.""")
parser.add_argument('--verbose', default=False, type=int, help='verbose level of logging.')
parser.add_argument('--max_workers', default=4, type=int, help='the number of workers in parrallel calculation.')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args.frontend)
    print(args.root)
    # real_micro_service test suite
    # print("\n{:!^80}\n".format(" Real Micro Service Test Suite "))

    # data_source = "real_micro_service"
    # entry_point_list = [14]
    # true_root_cause = [6, 28, 30, 31]
    # verbose = False
    # max_workers = 4
    # granger causal interval extend test
    tic = time.time()
    test_dycause(
        # Data params
        data_source=args.data_source,
        aggre_delta=args.agg,
        start_time=args.start,
        before_length=args.bef,
        after_length=args.aft,
        # Granger interval based graph construction params
        step=args.step,
        significant_thres=args.thres,
        lag=args.lag,  # must satisfy: step > 3 * lag + 1
        auto_threshold_ratio=args.edge_thres,
        runtime_debug=args.debug,
        # Root cause analysis params
        testrun_round=args.testrun,
        frontend=args.frontend,
        true_root_cause=args.root,
        max_path_length=args.max_path,
        mean_method=args.mean,
        topk_path=args.topk,
        num_sel_node=args.num_sel,
        # Debug params
        plot_figures=args.plot,
        verbose=args.verbose,
        max_workers=args.max_workers,
    )
    toc = time.time() - tic
    print('Used time: {:.4f} seconds'.format(toc))
