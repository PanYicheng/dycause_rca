# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 19:48:18 2016
"""

# causality.py
import numpy as np

import time
import datetime
import timeit

from statsmodels.tsa.stattools import grangercausalitytests as granger_std
from statsmodels.tsa.stattools import adfuller as adfuller

# import output_tools

import sys

from scipy import stats
from statsmodels.tsa.stattools import lagmat2ds as lagmat2ds
from statsmodels.tools.tools import add_constant as add_constant
from statsmodels.regression.linear_model import OLS as OLS


def test_granger(array_data, array_data_head, feature, target, lag,
                 significant_thres):
    """
    运行单个数据上的Grange test
    参数：
        array_data_head: 数据的header list
        feature: feature的名称
        target: target的名称
    """

    n_sample = len(array_data)

    print('sample size: ' + str(n_sample))

    print(feature)

    time1 = timeit.default_timer()

    index_target = array_data_head.index(target)
    array_target = array_data[:, index_target:index_target + 1].astype(float)

    index_feature = array_data_head.index(feature)
    array_feature = array_data[:, index_feature:index_feature +
                               1].astype(float)

    array_YX = np.concatenate((array_target, array_feature), axis=1)
    array_XY = np.concatenate((array_feature, array_target), axis=1)

    results_YX = granger_std(array_YX[:, :], lag, addconst=True,
                             verbose=False)[lag]
    results_XY = granger_std(array_XY[:, :], lag, addconst=True,
                             verbose=False)[lag]

    pvalue_YX = results_YX[0]['ssr_ftest'][1]
    pvalue_XY = results_XY[0]['ssr_ftest'][1]

    adfstat_Y, pvalue_Y, usedlag_Y, nobs_Y, critvalues_Y, icbest_Y = adfuller(
        array_target[:, 0], lag)
    adfstat_X, pvalue_X, usedlag_X, nobs_X, critvalues_X, icbest_X = adfuller(
        array_feature[:, 0], lag)

    if pvalue_Y < significant_thres and pvalue_X < significant_thres:

        if pvalue_YX < significant_thres and pvalue_XY > significant_thres:
            print(feature + ' causes ' + target)
        else:
            if pvalue_YX < significant_thres:
                print(feature + '->' + target)
            if pvalue_XY < significant_thres:
                print(target + '->' + feature)
            if pvalue_YX >= significant_thres and pvalue_XY >= significant_thres:
                print('no causality')

    else:
        print('not stationary')

    time2 = timeit.default_timer()

    print('total time: ' + str(time2 - time1))


class cnts_prune:

    def __init__(self, cnt_promising, cnt_promising_not, cnt_not_sure,
                 cnt_initial):
        self.cnt_promising = cnt_promising
        self.cnt_promising_not = cnt_promising_not
        self.cnt_not_sure = cnt_not_sure
        self.cnt_initial = cnt_initial

    def __str__(self):
        return ('Promising: %05d, PromisingNot: %05d, NotSure: %05d, Initial: %05d'
                % (self.cnt_promising, self.cnt_promising_not,
                   self.cnt_not_sure, self.cnt_initial))


def pick_a_date(array_data, array_data_head, date):
    """
    选择指定日期的数据
    """
    index_date_d = array_data_head.index('date_d')
    array_data = array_data[np.where(
        array_data[:, index_date_d].astype(int) == int(date))[0]]
    return array_data


def pick_a_trip(array_data, trip, list_dividing_limit):
    """
    从数据中选取索引在[ list_dividing_limit[trip] : list_dividing_limit[trip+1] ]
    
    的数据
    """
    start = list_dividing_limit[trip]
    end = list_dividing_limit[trip + 1]

    return array_data[start:end, :]


def divide_the_trip(array_data, array_data_head, time_diff_thres):
    """
    从数据中找出时间间隔大于阈值的数据索引

    Paramters:
        array_data: sample data (each row is a sample)
        array_data_head: headers of every column in array_data
        time_diff_thres: the lower bound of time difference extracted (seconds)
    Returns:
        list_dividing_limit: indices of the array_data which has time difference larger than time_diff_thres
        list_time_diff: the actual time differences
    """
    index_date_y = array_data_head.index('date_y')
    index_date_m = array_data_head.index('date_m')
    index_date_d = array_data_head.index('date_d')
    index_time_h = array_data_head.index('time_h')
    index_time_m = array_data_head.index('time_m')
    index_time_s = array_data_head.index('time_s')

    # new_array_data = np.copy(array_data)

    # transform times of every sample to a datetime list
    list_time = []
    for i in range(len(array_data)):
        sample = array_data[i]
        year = int(sample[index_date_y])
        month = int(sample[index_date_m])
        day = int(sample[index_date_d])
        hour = int(sample[index_time_h])
        minute = int(sample[index_time_m])
        second = int(sample[index_time_s])
        list_time.append(
            datetime.datetime(year, month, day, hour, minute, second))

    array_time = np.array(list_time)

    # new_array_time = np.copy(array_time)
    # list with indices in array_data where time difference between adjacent indices is larger than time_diff_thres
    list_dividing_limit = []
    # list of the actual time differences w.r.t. above list
    list_time_diff = [0]

    for sample_cnt in range(len(array_time) - 1):
        time_diff = (array_time[sample_cnt + 1] -
                     array_time[sample_cnt]).seconds
        if time_diff > time_diff_thres:
            list_dividing_limit.append(sample_cnt + 1)
            list_time_diff.append(time_diff)

    list_dividing_limit.insert(0, 0)
    list_dividing_limit.append(len(array_time))

    return list_dividing_limit, list_time_diff


def interpolation(array_data, array_data_head):
    """
    按照3秒时间间隔对数据进行线性插值补全数据
    """
    interpolation_time_step = 3

    index_date_y = array_data_head.index('date_y')
    index_date_m = array_data_head.index('date_m')
    index_date_d = array_data_head.index('date_d')
    index_time_h = array_data_head.index('time_h')
    index_time_m = array_data_head.index('time_m')
    index_time_s = array_data_head.index('time_s')

    list_time = []
    for i in range(len(array_data)):
        sample = array_data[i]
        year = int(sample[index_date_y])
        month = int(sample[index_date_m])
        day = int(sample[index_date_d])
        hour = int(sample[index_time_h])
        minute = int(sample[index_time_m])
        second = int(sample[index_time_s])
        list_time.append(
            datetime.datetime(year, month, day, hour, minute, second))

    array_time = np.array(list_time)

    new_list_time = [array_time[0]]

    new_list_data = [array_data[0]]

    # start interpolation

    new_sample_cnt = 0
    sample_cnt = 1

    while new_list_time[-1] < array_time[-1]:
        if sample_cnt == len(array_time):
            break
        time_diff = (array_time[sample_cnt] -
                     new_list_time[new_sample_cnt]).seconds
        if time_diff == interpolation_time_step:
            new_list_time.append(array_time[sample_cnt])
            new_list_data.append(array_data[sample_cnt])
            new_sample_cnt += 1
            sample_cnt += 1

        elif time_diff < interpolation_time_step:
            sample_cnt += 1
        elif time_diff > interpolation_time_step:
            new_list_time.append(new_list_time[new_sample_cnt] +
                                 datetime.timedelta(0, 3))
            new_sample_cnt += 1

            data_step = (array_data[sample_cnt].astype(float) -
                         array_data[sample_cnt - 1].astype(float)) / (
                             (array_time[sample_cnt] -
                              array_time[sample_cnt - 1]).seconds)
            this_data = array_data[sample_cnt - 1] + (
                new_list_time[new_sample_cnt] -
                array_time[sample_cnt - 1]).seconds * data_step
            new_list_data.append(this_data)

    new_array_data = np.array(new_list_data)
    # new_array_time = np.array(new_list_time)

    for sample_cnt in range(len(new_array_data)):
        for j in range(len(new_array_data[sample_cnt])):
            new_array_data[sample_cnt, j] = round(
                new_array_data[sample_cnt, j], 5)

    return new_array_data


def loop_granger(array_data, array_data_head, path_to_output, feature, target,
                 significant_thres, test_mode, trip, lag, step, simu_real,
                 max_segment_len, min_segment_len, verbose=True, return_result=False):
    """
    Granger causal intervals论文的优化算法
    参数：
        array_data: 时间序列数据，每列为一个变量
        array_data_head: 变量的名称
        path_to_output:
        feature: Granger causality的源变量名称 (feature -> target)
        target: Granger causality的目标变量名称 (feature -> target)
        significant_thres: 假设检验的显著性水平
        test_mode: 不同优化实现方式，这里默认使用最好的优化方式，即fast_version_3
        trip: 选择时间序列的哪个时间段，只在simu_real为real时有效
        lag: Granger causality test的最大历史间隔
        step: 划分区间的最小步长
        simu_real: 是否为模拟或真实数据，在真实数据下将会作适当的数据插值
        max_segment_len:进行因果检验的最大区间长度
        min_segment_len:进行因果检验的最小区间长度
        verbose: whether print detailed info
        return_result: whether return result p value matrix
    """
    #    sys.stdout = open(path_to_output+'log', 'w')

    addconst = True
    # in seconds
    time_diff_thres = 60
    # 至少要包含的数据样本数，小于则不做测试
    min_trip_length = 20

    #    output_tools.output_2d_data(array_data, array_data_head,
    #    path_to_output, str(trip) +'_before_interpolation'+'.csv')

    if simu_real == 'real':
        list_dividing_limit, list_time_diff = divide_the_trip(
            array_data, array_data_head, time_diff_thres)

        index_date_d = array_data_head.index('date_d')
        # list_segment = [[
        #     i, array_data[list_dividing_limit[i], index_date_d],
        #     list_dividing_limit[i], list_dividing_limit[i + 1],
        #     list_dividing_limit[i + 1] - list_dividing_limit[i],
        #     list_time_diff[i]
        # ] for i in range(len(list_dividing_limit) - 1)]
        # output_tools.output_2d_data(list_segment, [
        #     'segment_id', 'date', 'start', 'end', 'length',
        #     'time_diff_with_pre'
        # ], path_to_output, 'trip_division.csv')

        array_data = pick_a_trip(array_data, trip,
                                 list_dividing_limit)

        if len(array_data) < min_trip_length:
            return 0
        #array_data = pick_a_date(array_data, array_data_head, date)
        #index_hour = array_data_head.index('time_h')
        #array_data = array_data[np.where(array_data[:,index_hour] <= 18)[0]]
        # 通过插值使得数据样本间隔3秒
        array_data = interpolation(array_data, array_data_head)

        # output_tools.output_2d_data(
        #     array_data, array_data_head, path_to_output,
        #     str(trip) + '_after_interpolation' + '.csv')
    elif simu_real == 'simu':
        if verbose:
            print('{:-^60}'.format('Simulation'))

    n_sample = len(array_data)

    if verbose:
        print ('Sample size: ' + str(n_sample))

    #    file_para = open(path_to_output+'parameters', 'w')
    #    file_para.write('trip: '+str(trip)+'\n')
    #    file_para.write('lag: '+str(lag)+'\n')
    #    file_para.write('step: '+str(step)+'\n')
    #    file_para.write('min_segment_len: '+str(min_segment_len)+'\n')
    #    file_para.write('significant_thres: '+str(significant_thres)+'\n')
    #    file_para.write('test mode: '+str(test_mode)+'\n')
    #    file_para.write('target: '+str(target)+'\n')
    #    file_para.write('add const: '+str(addconst)+'\n')
    #    file_para.write('time diff thres: '+str(time_diff_thres)+'\n')
    #    file_para.close()

    fea_cnt = 0

    cnt_prune_YX = cnts_prune(0, 0, 0, 0)
    cnt_prune_XY = cnts_prune(0, 0, 0, 0)

    if verbose:
        print('Feature: \n',' '*5, feature)

    time1 = timeit.default_timer()

    time_granger = 0
    time_adf = 0

    index_target = array_data_head.index(target)
    array_target = array_data[:, index_target:index_target + 1].astype(float)

    index_feature = array_data_head.index(feature)
    array_feature = array_data[:, index_feature:index_feature +
                               1].astype(float)

    array_YX = np.concatenate((array_target, array_feature), axis=1)
    array_XY = np.concatenate((array_feature, array_target), axis=1)

    # begin loop

    n_step = int(n_sample / step)
    list_segment_split = [step * i for i in range(n_step)]
    if n_sample > step * (n_step):
        list_segment_split.append(n_sample)
    else:
        list_segment_split.append(step * n_step)

    start = 0
    end = 0

    total_cnt_segment_YX = 0
    total_cnt_segment_XY = 0
    total_cnt_segment_adf = 0
    total_cnt_segment_cal_adf = 0
    total_cnt_segment_examine_adf_Y = 0

    array_results_YX = np.full((n_step + 1, n_step + 1), -2, dtype=float)
    array_results_XY = np.full((n_step + 1, n_step + 1), -2, dtype=float)

    array_adf_results_X = np.full((n_step + 1, n_step + 1), -2, dtype=float)
    array_adf_results_Y = np.full((n_step + 1, n_step + 1), -2, dtype=float)

    array_res2down_ssr_YX = np.full((n_step + 1, n_step + 1), -2, dtype=float)
    array_res2djoint_ssr_YX = np.full((n_step + 1, n_step + 1),
                                      -2,
                                      dtype=float)

    for i in range(n_step):
        start = list_segment_split[i]
        # print (str(start) + '/' + str(len(array_YX)))

        reset_cnt_YX = -1
        res2down_YX = None
        res2djoint_YX = None
        res2down_ssr_upper_YX = 0
        res2down_ssr_lower_YX = 0
        res2djoint_ssr_upper_YX = 0
        res2djoint_ssr_lower_YX = 0
        res2djoint_df_resid_YX = 0

        reset_cnt_XY = -1
        res2down_XY = None
        res2djoint_XY = None
        res2down_ssr_upper_XY = 0
        res2down_ssr_lower_XY = 0
        res2djoint_ssr_upper_XY = 0
        res2djoint_ssr_lower_XY = 0
        res2djoint_df_resid_XY = 0

        for j in range(i + 1, n_step + 1):
            end = list_segment_split[j]
            # print ('Interval: [%d, %d]' % (start, end), end='')
            # 如果分段长度过小或者过大都跳过因果检验
            if len(array_YX[start:end, :]) < min_segment_len \
                    or len(array_YX[start:end, :]) > max_segment_len:
                array_results_YX[i, j] = -2
                array_results_XY[i, j] = -2
                array_adf_results_X[i, j] = -2
                array_adf_results_Y[i, j] = -2
                array_res2down_ssr_YX[i, j] = -2
                array_res2djoint_ssr_YX[i, j] = -2
                # print(' Skipped')
                continue
            # print(' Checked')
            time3 = timeit.default_timer()
            # granger test (call package)
            if test_mode == 'call_package':
                result = granger_std(array_YX[start:end, :],
                                     lag,
                                     addconst=True,
                                     verbose=False)
                p_value_YX = result[5][0]['ssr_ftest'][1]
                if p_value_YX < significant_thres:
                    result = granger_std(array_XY[start:end, :],
                                         lag,
                                         addconst=True,
                                         verbose=False)
                    p_value_XY = result[5][0]['ssr_ftest'][1]
                else:
                    p_value_XY = -1
                array_results_YX[i, j] = p_value_YX
                array_results_XY[i, j] = p_value_XY
            # granger test (standard)
            elif test_mode == 'standard':
                p_value_YX, res2down_YX, res2djoint_YX = grangercausalitytests(
                    array_YX[start:end, :], lag, addconst=True, verbose=False)
                if p_value_YX < significant_thres:
                    p_value_XY, res2down_XY, res2djoint_XY \
                        = grangercausalitytests(
                            array_XY[start:end, :],
                            lag,
                            addconst=True,
                            verbose=False)
                else:
                    p_value_XY = -1
                array_results_YX[i, j] = p_value_YX
                array_results_XY[i, j] = p_value_XY
            # granger test (fast) only check F_upper
            elif test_mode == 'fast_version_1':
                #    print 'array_YX shape:'
                #    print np.shape(array_YX[start:end,:])
                (p_value_YX, res2down_YX, res2djoint_YX,
                 res2down_ssr_upper_YX, res2djoint_ssr_lower_YX,
                 res2djoint_df_resid_YX, reset_cnt_YX) = \
                    grangercausalitytests_check_F_upper(
                        array_YX[start:end, :],
                        lag,
                        res2down_YX,
                        res2djoint_YX,
                        res2down_ssr_upper_YX,
                        res2djoint_ssr_lower_YX,
                        res2djoint_df_resid_YX,
                        significant_thres,
                        step,
                        reset_cnt_YX,
                        addconst=True,
                        verbose=False,
                    )
                if p_value_YX < significant_thres and p_value_YX != -1:
                    (p_value_XY, res2down_XY,
                    res2djoint_XY) = grangercausalitytests(
                        array_XY[start:end, :], lag, addconst, verbose=False)
                else:
                    p_value_XY = -1
                array_results_YX[i, j] = p_value_YX
                array_results_XY[i, j] = p_value_XY

                array_res2down_ssr_YX[i, j] = res2down_ssr_upper_YX
                array_res2djoint_ssr_YX[i, j] = res2djoint_ssr_lower_YX
            # check F_upper then check F_lower
            elif test_mode == 'fast_version_2':
                # granger test (fast)
                #    print 'array_YX shape:'
                #    print np.shape(array_YX[start:end,:])
                total_cnt_segment_YX += 1

                (p_value_YX, res2down_YX, res2djoint_YX,
                 res2down_ssr_upper_YX, res2down_ssr_lower_YX,
                 res2djoint_ssr_upper_YX, res2djoint_ssr_lower_YX,
                 res2djoint_df_resid_YX, reset_cnt_YX, cnt_prune_YX) \
                    = grangercausalitytests_check_F_upper_lower(
                        array_YX[start:end, :], lag, res2down_YX,
                        res2djoint_YX,
                        res2down_ssr_upper_YX, res2down_ssr_lower_YX,
                        res2djoint_ssr_upper_YX, res2djoint_ssr_lower_YX,
                        res2djoint_df_resid_YX, significant_thres, step,
                        reset_cnt_YX, cnt_prune_YX,
                        addconst=True, verbose=False)

                if p_value_YX < significant_thres and p_value_YX != -1:
                    total_cnt_segment_XY += 1
                    p_value_XY, res2down_XY, res2djoint_XY = \
                        grangercausalitytests(
                            array_XY[start:end, :], lag, addconst,
                            verbose=False)
                else:
                    p_value_XY = -1
                array_results_YX[i, j] = p_value_YX
                array_results_XY[i, j] = p_value_XY

                array_res2down_ssr_YX[i, j] = res2down_ssr_upper_YX
                array_res2djoint_ssr_YX[i, j] = res2djoint_ssr_lower_YX
            # check YX then check XY
            elif test_mode == 'fast_version_3':
                # granger test (fast)
                #    print 'array_YX shape:'
                #    print np.shape(array_YX[start:end,:])
                total_cnt_segment_YX += 1

                (p_value_YX, res2down_YX, res2djoint_YX,
                 res2down_ssr_upper_YX, res2down_ssr_lower_YX,
                 res2djoint_ssr_upper_YX, res2djoint_ssr_lower_YX,
                 res2djoint_df_resid_YX, reset_cnt_YX, cnt_prune_YX) \
                    = grangercausalitytests_check_F_upper_lower(
                        array_YX[start:end, :], lag, res2down_YX,
                        res2djoint_YX,
                        res2down_ssr_upper_YX, res2down_ssr_lower_YX,
                        res2djoint_ssr_upper_YX, res2djoint_ssr_lower_YX,
                        res2djoint_df_resid_YX, significant_thres, step,
                        reset_cnt_YX, cnt_prune_YX,
                        addconst=True, verbose=False)

                if p_value_YX < significant_thres and p_value_YX != -1:
                    total_cnt_segment_XY += 1
                    (p_value_XY, res2down_XY, res2djoint_XY,
                     res2down_ssr_upper_XY, res2down_ssr_lower_XY,
                     res2djoint_ssr_upper_XY, res2djoint_ssr_lower_XY,
                     res2djoint_df_resid_XY, reset_cnt_XY, cnt_prune_XY) \
                        = grangercausalitytests_check_F_upper_lower(
                            array_XY[start:end, :], lag, res2down_XY,
                            res2djoint_XY,
                            res2down_ssr_upper_XY, res2down_ssr_lower_XY,
                            res2djoint_ssr_upper_XY, res2djoint_ssr_lower_XY,
                            res2djoint_df_resid_XY, significant_thres, step,
                            reset_cnt_XY, cnt_prune_XY,
                            addconst=True, verbose=False)
                else:
                    p_value_XY = -1

                    if res2down_XY is not None and res2djoint_XY is not None:
                        res2down_ssr_upper_XY, res2down_ssr_lower_XY,
                        res2djoint_ssr_upper_XY, res2djoint_ssr_lower_XY,
                        res2djoint_df_resid_XY = update_bound(
                            array_XY[start:end, :],
                            res2down_XY,
                            res2djoint_XY,
                            res2down_ssr_upper_XY,
                            res2down_ssr_lower_XY,
                            res2djoint_ssr_upper_XY,
                            res2djoint_ssr_lower_XY,
                            lag,
                            step)

                        if res2down_XY.ssr > res2down_ssr_upper_XY \
                                or res2djoint_XY.ssr > res2djoint_ssr_upper_XY:
                            print('error')

                array_results_YX[i, j] = p_value_YX
                array_results_XY[i, j] = p_value_XY

                array_res2down_ssr_YX[i, j] = res2down_ssr_upper_YX
                array_res2djoint_ssr_YX[i, j] = res2djoint_ssr_lower_YX

            time4 = timeit.default_timer()

            time_granger += (time4 - time3)

            # stationary test

            time5 = timeit.default_timer()

            total_cnt_segment_adf += 1

            if p_value_YX < significant_thres and p_value_YX != -1 \
                    and p_value_XY > significant_thres:

                total_cnt_segment_examine_adf_Y += 1

                (adfstat_Y, pvalue_Y, usedlag_Y, nobs_Y, critvalues_Y,
                 icbest_Y) = adfuller(array_XY[start:end, 1], lag)
                array_adf_results_Y[i, j] = pvalue_Y

                if pvalue_Y < significant_thres:

                    (adfstat_X, pvalue_X, usedlag_X, nobs_X, critvalues_X,
                     icbest_X) = adfuller(array_XY[start:end, 0], lag)
                    array_adf_results_X[i, j] = pvalue_X
                    total_cnt_segment_cal_adf += 1

                else:

                    pvalue_X = -1
                    array_adf_results_X[i, j] = pvalue_X

            else:
                pvalue_Y = -1
                pvalue_X = -1
                array_adf_results_Y[i, j] = pvalue_Y
                array_adf_results_X[i, j] = pvalue_X

            # reject the granger result

            if pvalue_Y > significant_thres or pvalue_X > significant_thres:
                array_results_YX[i, j] = -1
                array_results_XY[i, j] = -1

            time6 = timeit.default_timer()

            time_adf += (time6 - time5)

    time2 = timeit.default_timer()

    # print ('total time: ' + str(time2 - time1))

    total_time = time2 - time1

    #    file_time = open(path_to_output+'time', 'a+')
    #    file_time.write(feature+',total time: ' + str(time2 - time1)+'\n')
    #    file_time.write(feature+',granger time: '+ str(time_granger)+'\n')
    #    file_time.write(feature+',adf time: '+ str(time_adf)+'\n')
    #    file_time.write('\n')
    #    file_time.close()
    #
    #
    #    if test_mode == 'fast_version_2' or test_mode == 'fast_version_3':
    #        file_cnts_prune = open(path_to_output+'prune_cnts', 'a+')
    #        file_cnts_prune.write(feature+',for YX:\n')
    #        file_cnts_prune.write(feature+',total cnt: ' + str(total_cnt_segment_YX)+'\n')
    #        file_cnts_prune.write(feature+',promising cnt: '+ str(cnt_prune_YX.cnt_promising)+'\n')
    #        file_cnts_prune.write(feature+',promising not cnt: '+ str(cnt_prune_YX.cnt_promising_not)+'\n')
    #        file_cnts_prune.write(feature+',not sure cnt: '+ str(cnt_prune_YX.cnt_not_sure)+'\n')
    #        file_cnts_prune.write(feature+',initial cnt: '+ str(cnt_prune_YX.cnt_initial)+'\n')
    #        file_cnts_prune.write(feature+',for XY:\n')
    #        file_cnts_prune.write(feature+',total cnt: ' + str(total_cnt_segment_XY)+'\n')
    #        file_cnts_prune.write(feature+',promising cnt: '+ str(cnt_prune_XY.cnt_promising)+'\n')
    #        file_cnts_prune.write(feature+',promising not cnt: '+ str(cnt_prune_XY.cnt_promising_not)+'\n')
    #        file_cnts_prune.write(feature+',not sure cnt: '+ str(cnt_prune_XY.cnt_not_sure)+'\n')
    #        file_cnts_prune.write(feature+',initial cnt: '+ str(cnt_prune_XY.cnt_initial)+'\n')
    #        file_cnts_prune.write(feature+',total adf cnt: '+ str(total_cnt_segment_adf)+'\n')
    #        file_cnts_prune.write(feature+',total cal adf cnt: '+ str(total_cnt_segment_cal_adf)+'\n')
    #        file_cnts_prune.write(feature+', total_cnt_segment_examine_adf_Y: '+str(total_cnt_segment_examine_adf_Y)+'\n')
    #        file_cnts_prune.write('\n')
    #        file_cnts_prune.close()
    #
    #
    #    output_tools.output_2d_data(array_results_YX, [], path_to_output, str(trip) +'_'+target+'_caused_by_'+feature+'.csv')
    #    output_tools.output_2d_data(array_results_XY, [], path_to_output, str(trip) +'_'+feature+'_caused_by_'+target+'.csv')
    if not return_result:
        np.save(path_to_output+(target+'_caused_by_'+feature).replace('/', '-'), array_results_YX)
        np.save(path_to_output+(feature+'_caused_by_'+target).replace('/', '-'), array_results_XY)
    #
    #
    #    output_tools.output_2d_data(array_adf_results_X, [], path_to_output, str(trip) +'_'+feature+'_adf'+'.csv')
    #
    #
    #    output_tools.output_2d_data(array_adf_results_Y, [], path_to_output, str(trip) +'_'+target+'_with_'+feature+'_adf'+'.csv')
    #
    #    output_tools.output_2d_data(array_res2down_ssr_YX, [], path_to_output, str(trip) +'_'+target+'_caused_by_'+feature+'_res2down_ssr.csv')
    #    output_tools.output_2d_data(array_res2djoint_ssr_YX, [], path_to_output, str(trip) +'_'+target+'_caused_by_'+feature+'_res2djoint_ssr.csv')
    #
    #    output_tools.output_list(list_segment_split, path_to_output, 'segment_split_id.csv')

    fea_cnt += 1
    if verbose:
        print('Prune for Y <-- X:\n', ' '*5, cnt_prune_YX)
        print('Prune for X <-- Y:\n', ' '*5, cnt_prune_XY)
    if return_result:
        return total_time, time_granger, time_adf, array_results_YX, array_results_XY
    return total_time, time_granger, time_adf


def get_lagged_data(x, lag, addconst, verbose):
    """
    生成lag矩阵

    对于x=[Y X], 生成每一行包含 [Y_t Y_{t-1} Y_{t-2} ... Y_{t-lag} X_{t-1} X_{t-2} ... X_{t-lag}] 的数据

    Returns:
        dta: 整个lag矩阵
        dtaown: 只包含自己过去lag的矩阵, 即[Y_{t-1} Y_{t-2} ... Y_{t-lag}]
        dtajoint: 包含自己和其他变量过去lag时刻的数据，即 [Y_{t-1} Y_{t-2} ... Y_{t-lag} X_{t-1} X_{t-2} ... X_{t-lag}]
     """
    x = np.asarray(x)
    #    print x.shape[0]
    #    print x.shape
    if x.shape[0] <= 3 * lag + int(addconst):
        raise ValueError(
            "Insufficient observations. Maximum allowable "
            "lag is {0}".format(int((x.shape[0] - int(addconst)) / 3) - 1))
    if verbose:
        print('\nGranger Causality')
        print('number of lags (no zero)', lag)
    # create lagmat of both time series
    """
    例如对于列变量为[Y_t, X_t]的数据，调用下面语句之后会列变量变成
    [Y_t Y_{t-1} Y_{t-2} ... Y_{t-lag} X_{t-1} X_{t-2} ... X_{t-lag}] 
     """
    dta = lagmat2ds(x, lag, trim='both', dropex=1)
    # add constant
    if addconst:
        dtaown = add_constant(dta[:, 1:(lag + 1)],
                              prepend=False,
                              has_constant='add')
        dtajoint = add_constant(dta[:, 1:], prepend=False, has_constant='add')
    else:
        dtaown = dta[:, 1:(lag + 1)]
        dtajoint = dta[:, 1:]
    return dta, dtaown, dtajoint


def fit_regression(dta, dtaown, dtajoint):
    """
    针对部分模型和全模型进行两次线性拟合，并返回结果  
     """
    # Run ols on both models without and with lags of second variable
    res2down = OLS(dta[:, 0], dtaown).fit()
    res2djoint = OLS(dta[:, 0], dtajoint).fit()

    #print results
    #for ssr based tests see:
    #http://support.sas.com/rnd/app/examples/ets/granger/index.htm
    #the other tests are made-up

    return res2down, res2djoint


def f_test(res2down, res2djoint, lag):
    """ 
     根据拟合结果进行F统计检验，返回统计值

     Returns: a dict {'ssr_ftest':
                            (F-statistics, 
                            stats.f.sf(fgc1, lag, res2djoint.df_resid)(p_value),
                            res2djoint.df_resid(完全模型剩余自由度)), 
                            lag)
                         }
     """
    result = {}

    # Granger Causality test using ssr (F statistic)
    # TODO: possible divide by 0
    fgc1 = (res2down.ssr -
            res2djoint.ssr) / res2djoint.ssr / lag * res2djoint.df_resid

    #    if verbose:
    #        print('ssr based F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,'
    #               ' df_num=%d' % (fgc1,
    #                                stats.f.sf(fgc1, lag,
    #                                           res2djoint.df_resid),
    #                                res2djoint.df_resid, lag))
    result['ssr_ftest'] = (fgc1, stats.f.sf(fgc1, lag, res2djoint.df_resid),
                           res2djoint.df_resid, lag)

    return result


def grangercausalitytests(x, lag, addconst=True, verbose=False):
    """
     采用自定义的方法进行Granger causality检验，只对lag进行f test。

     而statsmodels里的方法会对从1到lag的间隔都采取4种假设检验，效率较低。
     """
    dta, dtaown, dtajoint = get_lagged_data(x, lag, addconst, verbose)

    res2down, res2djoint = fit_regression(dta, dtaown, dtajoint)

    result = f_test(res2down, res2djoint, lag)

    p_value = result['ssr_ftest'][1]

    return p_value, res2down, res2djoint


def update_bound(x, pre_res2down, pre_res2djoint, pre_res2down_ssr_upper,
                 pre_res2down_ssr_lower, pre_res2djoint_ssr_upper,
                 pre_res2djoint_ssr_lower, lag, step,
                 addconst=True, verbose=False):
    """ 
     根据上一步的ssr计算当前这一步的ssr的上下界

     Parameters:
        x:用于补充未知拟合点的误差的数据
        pre_res2down:   前一步半模型结果
        pre_res2djoint: 前一步全模型结果
        pre_res2down_ssr_upper:
        pre_res2down_ssr_lower:
        pre_res2djoint_ssr_upper:
        pre_res2djoint_ssr_lower:
        pre_res2djoint_df_resid: 全模型的自由度
        lag:
        step: 每步使用样本量，用于确定补充误差需要使用的样本数量
        addconst, verbose: 用于产生lag数据
     """
    dta, dtaown, dtajoint = get_lagged_data(x, lag, addconst, verbose)
    res2down_fit_new_point_error = np.dot(dtaown[-step:, :],
                                          pre_res2down.params) - dta[-step:, 0]
    res2djoint_fit_new_point_error = np.dot(
        dtajoint[-step:, :], pre_res2djoint.params) - dta[-step:, 0]

    res2down_ssr_upper = np.dot(
        res2down_fit_new_point_error,
        res2down_fit_new_point_error) + pre_res2down_ssr_upper
    res2djoint_ssr_lower = pre_res2djoint_ssr_lower
    res2down_ssr_lower = pre_res2down_ssr_lower
    res2djoint_ssr_upper = np.dot(
        res2djoint_fit_new_point_error,
        res2djoint_fit_new_point_error) + pre_res2djoint_ssr_upper

    non_zero_column = np.sum(np.sum(dtajoint[:, :], axis=0) != 0)
    res2djoint_df_resid = len(dtajoint) - non_zero_column

    return (res2down_ssr_upper, res2down_ssr_lower,
            res2djoint_ssr_upper, res2djoint_ssr_lower, 
            res2djoint_df_resid)


def grangercausalitytests_check_F_upper_lower(x,
                                              lag,
                                              pre_res2down,
                                              pre_res2djoint,
                                              pre_res2down_ssr_upper,
                                              pre_res2down_ssr_lower,
                                              pre_res2djoint_ssr_upper,
                                              pre_res2djoint_ssr_lower,
                                              pre_res2djoint_df_resid,
                                              significant_thres,
                                              step,
                                              cnt,
                                              cnt_prune,
                                              addconst=True,
                                              verbose=False):

    dta, dtaown, dtajoint = get_lagged_data(x, lag, addconst, verbose)

    if cnt == -1:
        # initialization
        res2down, res2djoint = fit_regression(dta, dtaown, dtajoint)
        result = f_test(res2down, res2djoint, lag)
        p_value = result['ssr_ftest'][1]

        res2down_ssr_upper = res2down.ssr
        res2down_ssr_lower = res2down.ssr
        res2djoint_ssr_upper = res2djoint.ssr
        res2djoint_ssr_lower = res2djoint.ssr

        cnt_prune.cnt_initial += 1

        return (p_value, res2down, res2djoint, res2down_ssr_upper,
                res2down_ssr_lower, res2djoint_ssr_upper, res2djoint_ssr_lower,
                res2djoint.df_resid, 0, cnt_prune)
    else:
        # fit the new data
        # prune promising not
        res2down_fit_new_point_error = np.dot(
            dtaown[-step:, :], pre_res2down.params) - dta[-step:, 0]
        res2djoint_fit_new_point_error = np.dot(
            dtajoint[-step:, :], pre_res2djoint.params) - dta[-step:, 0]

        res2down_ssr_upper = np.dot(
            res2down_fit_new_point_error,
            res2down_fit_new_point_error) + pre_res2down_ssr_upper
        res2djoint_ssr_lower = pre_res2djoint_ssr_lower
        res2down_ssr_lower = pre_res2down_ssr_lower
        res2djoint_ssr_upper = np.dot(
            res2djoint_fit_new_point_error,
            res2djoint_fit_new_point_error) + pre_res2djoint_ssr_upper

        non_zero_column = np.sum(np.sum(dtajoint[:, :], axis=0) != 0)
        res2djoint_df_resid = len(dtajoint) - non_zero_column

        #        res2down, res2djoint = fit_regression(dta, dtaown, dtajoint)
        #        if (res2down_ssr_upper - res2down.ssr) < -0.00001 or
        #           (res2djoint_ssr_upper < res2djoint.ssr) < -0.00001:
        #            print 'error'
        #
        #        if res2djoint.df_resid != res2djoint_df_resid:
        #            print 'error'
        #
        # check F_upper
        # TODO: possible divide by 0
        F_upper = (res2down_ssr_upper / res2djoint_ssr_lower -
                   1) * (res2djoint_df_resid) / lag
        p_value_lower = 1 - stats.f.cdf(F_upper, lag,
                                              (res2djoint_df_resid))

        if p_value_lower >= significant_thres:  # promising not
            p_value = 1
            cnt_prune.cnt_promising_not += 1

            return (p_value, pre_res2down, pre_res2djoint,
                    res2down_ssr_upper, res2down_ssr_lower,
                    res2djoint_ssr_upper, res2djoint_ssr_lower,
                    res2djoint_df_resid, cnt + 1, cnt_prune)

        else:
            # check F_lower
            # TODO: possible divide by 0
            F_lower = (res2down_ssr_lower / res2djoint_ssr_upper -
                       1) * (res2djoint_df_resid) / lag
            p_value_upper = 1 - stats.f.cdf(F_lower, lag,
                                                  (res2djoint_df_resid))

            if p_value_upper < significant_thres:
                # promising
                p_value = 0
                cnt_prune.cnt_promising += 1
                return (p_value, pre_res2down, pre_res2djoint,
                        res2down_ssr_upper, res2down_ssr_lower,
                        res2djoint_ssr_upper, res2djoint_ssr_lower,
                        res2djoint_df_resid, cnt + 1, cnt_prune)

            else:
                # not sure
                res2down, res2djoint = fit_regression(dta, dtaown, dtajoint)
                result = f_test(res2down, res2djoint, lag)
                p_value = result['ssr_ftest'][1]

                res2down_ssr_upper = res2down.ssr
                res2down_ssr_lower = res2down.ssr
                res2djoint_ssr_upper = res2djoint.ssr
                res2djoint_ssr_lower = res2djoint.ssr

                cnt_prune.cnt_not_sure += 1

                return (p_value, res2down, res2djoint,
                        res2down_ssr_upper, res2down_ssr_lower,
                        res2djoint_ssr_upper, res2djoint_ssr_lower,
                        res2djoint.df_resid, cnt+1, cnt_prune)


def grangercausalitytests_check_F_upper(x,
                                        lag,
                                        pre_res2down,
                                        pre_res2djoint,
                                        pre_res2down_ssr_upper,
                                        pre_res2djoint_ssr_lower,
                                        pre_res2djoint_df_resid,
                                        significant_thres,
                                        step,
                                        cnt,
                                        addconst=True,
                                        verbose=False):

    dta, dtaown, dtajoint = get_lagged_data(x, lag, addconst, verbose)

    if cnt == -1:
        # initialization
        res2down, res2djoint = fit_regression(dta, dtaown, dtajoint)
        result = f_test(res2down, res2djoint, lag)
        p_value = result['ssr_ftest'][1]

        res2down_ssr_upper = res2down.ssr
        res2djoint_ssr_lower = res2djoint.ssr

        return (p_value, res2down, res2djoint, res2down_ssr_upper,
                res2djoint_ssr_lower, res2djoint.df_resid, 0)
    else:

        res2down_fit_new_point_error = np.dot(
            dtaown[-step:, :], pre_res2down.params) - dta[-step:, 0]
        res2down_ssr_upper = np.dot(
            res2down_fit_new_point_error,
            res2down_fit_new_point_error) + pre_res2down_ssr_upper
        res2djoint_ssr_lower = pre_res2djoint_ssr_lower

        non_zero_column = np.sum(np.sum(dtajoint[:, :], axis=0) != 0)
        res2djoint_df_resid = len(dtajoint) - non_zero_column

        F_upper = (res2down_ssr_upper / res2djoint_ssr_lower -
                   1) * (res2djoint_df_resid) / lag
        p_value_lower = 1 - stats.f.cdf(F_upper, lag,
                                              (res2djoint_df_resid))

        if p_value_lower < significant_thres:
            res2down, res2djoint = fit_regression(dta, dtaown, dtajoint)
            result = f_test(res2down, res2djoint, lag)
            p_value = result['ssr_ftest'][1]

            res2down_ssr_upper = res2down.ssr
            res2djoint_ssr_lower = res2djoint.ssr

            return (p_value, res2down, res2djoint, res2down_ssr_upper,
                    res2djoint_ssr_lower, res2djoint.df_resid, cnt+1)
        else:
            p_value = -1

            return (p_value, pre_res2down, pre_res2djoint, res2down_ssr_upper,
                    res2djoint_ssr_lower, res2djoint_df_resid, cnt+1)
