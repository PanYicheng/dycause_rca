import numpy as np
import sys
import pickle
from openpyxl import load_workbook
from main_dycause import test_dycause as test_granger_extend
import concurrent.futures
from concurrent.futures import as_completed
from tqdm import tqdm

# pymicro test suite
# dataset_name = 'pymicro'
# entry_point_list = [16]
# true_root_cause = [1]

# ibm_micro_service test suite
dataset_name = 'ibm_micro_service'
entry_point_list = [14]
true_root_cause = [6, 28, 30, 31]

params_list = []
for aggre_delta in range(1, 2):
    for before_length in [0]:
        for after_length in np.arange(0, 300, 20):
            for step in [50, 60, 70][0:1]:
                for sig_value in [0.1]:
                    for lag in [5, 10, 15][0:1]:
                        for thres in [0.5, 0.7, 0.9]:
                            for max_path_length in [None]:
                                for mean_method in ['arithmetic', 'geometric', 'harmonic'][2:3]:
                                    for topk_path in [150]:
                                        for num_sel_node in range(1, 4):
                                            if before_length != 0 or after_length != 0:
                                                params_list.append({
                                                    'ela': aggre_delta,
                                                    'bef': before_length, 
                                                    'aft': after_length, 
                                                    'step': step, 
                                                    'sig_value': sig_value,
                                                    'lag': lag, 
                                                    'thres': thres, 
                                                    'max_path_length': max_path_length,
                                                    'mean_method': mean_method,
                                                    'topk_path': topk_path,
                                                    'num_sel_node': num_sel_node
                                                    })
# for aggre_delta in range(1, 2):
#     for before_length in [100]:
#         for after_length in [0]:
#             for step in [30]:
#                 for sig_value in [0.1]:
#                     for lag in [5, 7, 9]:
#                         for thres in [0.5, 0.7, 0.9]:
#                             for max_path_length in [None]:
#                                 for mean_method in ['arithmetic', 'geometric', 'harmonic']:
#                                     for topk_path in [60]:
#                                         for num_sel_node in range(1, 2):
#                                             if before_length != 0 or after_length != 0:
#                                                 params_list.append({
#                                                     'ela': aggre_delta,
#                                                     'bef': before_length, 
#                                                     'aft': after_length, 
#                                                     'step': step, 
#                                                     'sig_value': sig_value,
#                                                     'lag': lag, 
#                                                     'thres': thres, 
#                                                     'max_path_length': max_path_length,
#                                                     'mean_method': mean_method,
#                                                     'topk_path': topk_path,
#                                                     'num_sel_node': num_sel_node
#                                                     })

def worker_process(ind, params_dict):
    prks, acc = test_granger_extend(
        # Data params
        data_source=dataset_name,
        aggre_delta=params_dict['ela'],
        start_time=4653,
        before_length=params_dict['bef'],
        after_length=params_dict['aft'],
        # Granger interval based graph construction params
        step=params_dict['step'],
        significant_thres=params_dict['sig_value'],
        lag=params_dict['lag'],
        auto_threshold_ratio = params_dict['thres'],
        # Root cause analysis params
        max_path_length=params_dict['max_path_length'],
        mean_method=params_dict['mean_method'],
        topk_path = params_dict['topk_path'],
        num_sel_node = params_dict['num_sel_node'],
        testrun_round=1,
        frontend=entry_point_list[0],
        true_root_cause=true_root_cause,
        # Debug params
        plot_figures=False,
        verbose=False,
        disable_print=True
    )
    return prks, acc, ind

def main():
    # print("\n{:!^80}\n".format(" Pymicro Test Suite "))
    print("\n{:!^80}\n".format(" IBM Micro Service Test Suite "))
    print('Number of parameter sets: {}'.format(len(params_list)))
    processbar = tqdm(total=len(params_list), ascii=True)
    futures = []
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=6)
    for i, params_dict in enumerate(params_list):
        futures.append(executor.submit(worker_process, i, params_dict))
    failed_task = 0
    result_list = []
    for future in as_completed(futures):
        processbar.update(1)
        try:
            prks, acc, ind = future.result()
            result_list.append(params_list[ind])
            result_list[-1]['prks'] = prks
            result_list[-1]['acc'] = acc
        except Exception as e:
            failed_task += 1
            print('Exception:{}, 1 task failed! Total failed: {}'
                    .format(e, failed_task))
    executor.shutdown()
    if len(result_list) > 0:
        with open('granger_extend_parameter_tune_ibm_708.pkl', 'wb') as f:
            pickle.dump(result_list, f)


if __name__ == '__main__':
    main()