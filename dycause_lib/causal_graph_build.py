import numpy as np


def normalize_by_column(transition_matrix):
    for col_index in range(transition_matrix.shape[1]):
        if np.sum(transition_matrix[:, col_index]) == 0:
            continue
        transition_matrix[:, col_index] = transition_matrix[:, col_index] / np.sum(
            transition_matrix[:, col_index]
        )
    return transition_matrix


def normalize_by_row(transition_matrix):
    for row_index in range(transition_matrix.shape[0]):
        if np.sum(transition_matrix[row_index, :]) == 0:
            continue
        transition_matrix[row_index, :] = transition_matrix[row_index, :] / np.sum(
            transition_matrix[row_index, :]
        )
    return transition_matrix


def get_overlay_count(n_sample, ordered_intervals):
    overlay_counts = np.zeros([n_sample, 1], dtype=np.int)
    for interval in ordered_intervals:
        overlay_counts[interval[1][0] : interval[1][1], 0] += 1
    return overlay_counts


def get_ordered_intervals(matrics, significant_thres, list_segment_split):
    array_results_YX, array_results_XY = matrics
    array_results_YX = np.abs(array_results_YX)
    array_results_XY = np.abs(array_results_XY)
    nrows, ncols = array_results_YX.shape
    intervals = []
    pvalues = []
    for i in range(nrows):
        for j in range(i + 1, ncols):
            if (abs(array_results_YX[i, j]) < significant_thres) and (
                array_results_XY[i, j] >= significant_thres
                or array_results_XY[i, j] == -1
            ):
                intervals.append((list_segment_split[i], list_segment_split[j]))
                pvalues.append((array_results_YX[i, j], array_results_XY[i, j]))
    ordered_intervals = list(zip(pvalues, intervals))
    ordered_intervals.sort(key=lambda x: (x[0][0], -x[0][1]))
    return ordered_intervals


def get_segment_split(n_sample, step):
    n_step = int(n_sample / step)
    list_segment_split = [step * i for i in range(n_step)]
    if n_sample > step * (n_step):
        list_segment_split.append(n_sample)
    else:
        list_segment_split.append(step * n_step)
    return list_segment_split


# Following are not used functions


def get_intervals_over_overlaythres(counts, overlay_thres):
    mask = counts > overlay_thres
    if not np.any(mask):
        return []
    indices = np.where(mask)[0]
    starts = [indices[0]]
    ends = []
    for i in np.where(np.diff(indices, axis=0) > 1)[0]:
        ends.append(indices[i] + 1)
        starts.append(indices[i + 1])
    ends.append(indices[-1] + 1)
    return list(zip(starts, ends))


def get_max_overlay_intervals(counts):
    if np.max(counts) == 0:
        return []
    sample_indices_max = np.where(np.max(counts) == counts)[0]
    starts = [sample_indices_max[0]]
    ends = []
    for i in np.where(np.diff(sample_indices_max, axis=0) > 1)[0]:
        ends.append(sample_indices_max[i] + 1)
        starts.append(sample_indices_max[i + 1])
    ends.append(sample_indices_max[-1] + 1)
    return list(zip(starts, ends))


def get_max_proportion(n_sample, ordered_intervals):
    x = np.zeros([n_sample])
    for interval in ordered_intervals:
        x[interval[1][0] : interval[1][1]] = 1
    return np.sum(x) / (0.0 + n_sample)
