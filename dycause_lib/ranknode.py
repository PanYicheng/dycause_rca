from collections import defaultdict

import numpy as np

from dycause_lib.randwalk import randwalk


def analyze_root(
    transition_matrix,
    entry_point,
    local_data,
    epoch=1000,
    mean_method="arithmetic",
    max_path_length=None,
    topk_path=60,
    prob_thres=0.4,
    num_sel_node=1,
    use_new_matrix=False,
    verbose=False,
):
    out_path, new_matrix = randwalk(
        transition_matrix,
        epoch=epoch,
        mean_method=mean_method,
        max_path_length=max_path_length,
        entry_point=entry_point,
        use_new_matrix=use_new_matrix,
        verbose=verbose,
    )
    ranked_nodes = ranknode(
        local_data,
        out_path,
        entry_point,
        local_data.shape[1],
        topk_path=topk_path,
        prob_thres=prob_thres,
        num_sel_node=num_sel_node,
    )
    # adjust node representation to data representation
    for j in range(len(ranked_nodes)):
        ranked_nodes[j][0] += 1
    return ranked_nodes, new_matrix


def ranknode(
    data, out_path, entry_point, node_num, topk_path=60, prob_thres=0.4, num_sel_node=1
):
    """Rank node according to pearson correlation
    """
    # region select X nodes from path
    path_node_count = defaultdict(int)
    # select only first topk paths with prob >= threshold
    for i in out_path[:topk_path]:
        for node in i[1][-num_sel_node:]:
            path_node_count[node] = path_node_count[node] + 1
        if i[0] < prob_thres:
            break
    # exclude entry point
    if entry_point - 1 in path_node_count:
        path_node_count.pop(entry_point - 1)
    # endregion

    # region Calculate correlation between selected node and entry point
    path_node_corr = {}
    for node in path_node_count:
        ret = np.corrcoef(
            np.concatenate(
                [data[:, entry_point - 1].reshape(1, -1), data[:, node].reshape(1, -1)],
                axis=0,
            )
        )
        #     print('Node:{} Corr:{}'.format(node, abs(ret[0, 1])))
        path_node_corr[node] = abs(ret[0, 1])
    # endregion

    # region Estimate node root cause score according to both
    #        path count and correlation
    rank_list = []
    for node in path_node_count:
        rank_list.append(
            [
                node,
                path_node_count[node] * 1.0 / (num_sel_node * topk_path)
                + path_node_corr[node] * 1.0,
            ]
        )
    rank_list.sort(key=lambda x: x[1], reverse=True)
    # endregion

    # append all other nodes according to correlation coefficient
    # other_node = []
    # for node in range(node_num):
    #     if node not in candidate_node:
    #         ret = np.corrcoef(
    #             np.concatenate(
    #                 [data[:, entry_point-1].reshape(1, -1),
    #                  data[:, node].reshape(1, -1)], axis=0))
    #         other_node.append([node, abs(ret[0, 1])])
    # other_node.sort(key=lambda x: x[1], reverse=True)
    # candidate.extend(other_node)
    return rank_list
