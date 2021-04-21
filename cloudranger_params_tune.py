import numpy as np
import sys
import pickle
from openpyxl import load_workbook
from main_cloud_ranger import test_cloud_ranger
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
for ela in range(1, 21):
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for testround in [5]:
            for beta in [0.1]:
                for rho in [0.2]:
                    params_list.append({'ela': ela, 'alpha': alpha, 'testround': testround, 
                                    'beta': beta, 'rho': rho})

def worker_process(ind, params_dict):
    prks, acc = test_cloud_ranger(
        data_source=dataset_name,
        pc_aggregate=params_dict['ela'],
        pc_alpha=params_dict['alpha'],
        testrun_round=params_dict['testround'],
        frontend=entry_point_list[0],
        true_root_cause=true_root_cause,
        beta=params_dict['beta'],
        rho=params_dict['rho'],
        save_data_fig=False,
        verbose=False,
        disable_print=True,
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
        with open('cloudranger_parameter_tune_ibm.pkl', 'wb') as f:
            pickle.dump(result_list, f)


if __name__ == '__main__':
    main()