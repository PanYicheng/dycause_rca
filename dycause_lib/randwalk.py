from collections import defaultdict

import numpy as np
from scipy import stats


def normalize_by_column(transition_matrix):
    for col_index in range(transition_matrix.shape[1]):
        if np.sum(transition_matrix[:, col_index]) == 0:
            continue
        transition_matrix[:, col_index] = transition_matrix[:, col_index] / np.sum(
            transition_matrix[:, col_index]
        )
    return transition_matrix


def geo_mean_overflow(iterable):
    a = np.log(iterable)
    return np.exp(a.sum() / len(a))


def bfs(
    transition_matrix, entry_point, reach_end=True, max_path_length=None, verbose=False
):
    """ Backtrace breadth first search in the graph.

    Params:
        transition_matrix: the transition matrix of the graph.
        entry_point: the entry point of the bfs, the frontend service.
        reach_end: if encouter the same node in seaching the path, stop if False; else search till no previous node exists.
        max_path_length: the maximum allowed length of bfs path, if None no limit is set.
        verbose: whether print detailed information.
    """
    path_list = set()
    queue = [[entry_point - 1]]
    while len(queue) > 0:
        # Limit output path list size to 10000 in case of infinite bfs
        if len(path_list) > 10000:
            break
        # Limit bfs queue size to 10000 in case of infinite bfs, and flush paths to path_list
        if len(queue) > 10000:
            while len(queue) > 0:
                path_list.add(tuple(queue.pop(0)))
            break
        if verbose and verbose>=2:
            # verbose level >= 2: print BFS queue info
            print(
                "{space:^15}BFS Queue Len:{i:>6d} Output queue size: {:>6d}"
                .format(len(path_list), space="", i=len(queue)),
                end="\r",
            )
            
        path = queue.pop(0)
        if np.sum(transition_matrix[:, path[-1]]) == 0:
            # if there is no previous node, the path ends and we add it to output.
            path_list.add(tuple(path))
        else:
            # if there is at least one previous node
            if max_path_length is not None and len(path) >= max_path_length:
                # if path length exceeds limit
                path_list.add(tuple(path))
            else:
                # Try extending the path with every possible node
                for prev_node in range(transition_matrix.shape[0]):
                    if transition_matrix[prev_node, path[-1]] > 0.0 and (
                        prev_node not in path
                    ):
                        # extend the path
                        new_path = path + [prev_node]
                        queue.append(new_path)
                    elif transition_matrix[prev_node, path[-1]] > 0.0 and not reach_end:
                        # if encounter repeated node, stop bfs if reach_end is set to False.
                        path_list.add(tuple(path))
    if verbose and verbose>=2:
        # verbose level >= 2: print BFS queue info endline
        print("")
    return path_list


def randwalk(
    transition_matrix,
    epoch=1000,
    mean_method="arithmetic",
    max_path_length=None,
    entry_point=29,
    use_new_matrix=False,
    verbose=True,
):
    """Rand walk using transition matrix and output all pathes sorted
    by path probability
    """
    # region Random walk to generate a more robust transition matrix
    if use_new_matrix:
        if verbose:
            print(
                "{space:^15}{name:<30}:".format(
                    space="", name="Randwalk to generate new matrix"
                )
            )
        node_num = transition_matrix.shape[0]
        node_list = range(node_num)
        node_visit = defaultdict(int)
        edge_visit = {}
        action_count = defaultdict(int)
        # action list: 0 -> father, 1 -> child, 2 -> stay
        action_list = range(3)
        for i in range(epoch):
            if verbose:
                print("{space:^20}Epoch:{i:>6d}".format(space="", i=i), end="\r")
            current_point = entry_point - 1
            # region Node visit
            # for j in range(max_path_length):
            #     action = np.random.choice(action_list, p=[0.8, 0.1, 0.1])
            #     action_count[action] += 1
            #     if verbose:
            #         # print('Epoch:{:>6d} node:{:>2d}'.format(
            #         #     i, current_point), end='\r')
            #         pass
            #     if action == 0:
            #         # Find father node
            #         if np.sum(transmission_matrix[:, current_point]) == 0:
            #             break
            #         father_node = np.random.choice(
            #             node_list,
            #             p=transmission_matrix[:, current_point] /
            #             np.sum(transmission_matrix[:, current_point]))
            #         current_point = father_node
            #     elif action == 1:
            #         # Find child node
            #         if np.sum(transmission_matrix[current_point, :]) == 0:
            #             break
            #         child_node = np.random.choice(
            #             node_list,
            #             p=transmission_matrix[current_point, :] /
            #             np.sum(transmission_matrix[current_point, :]))
            #         current_point = child_node
            #     elif action == 2:
            #         # Stay still
            #         pass
            #     else:
            #         pass
            #     node_visit[current_point] = node_visit[current_point]+1
            # endregion
            # region Edge visit
            for path in range(max_path_length):
                action = np.random.choice(range(3), p=[0.8, 0.1, 0.1])
                action_count[action] += 1
                if action == 0:
                    # Find father node
                    if np.sum(transition_matrix[:, current_point]) == 0:
                        break
                    source = np.random.choice(
                        range(transition_matrix.shape[0]),
                        p=transition_matrix[:, current_point]
                        / np.sum(transition_matrix[:, current_point]),
                    )
                    if (source, current_point) in edge_visit:
                        edge_visit[(source, current_point)] += 1
                    else:
                        edge_visit[(source, current_point)] = 1
                    current_point = source
                elif action == 1:
                    # Find child node
                    if np.sum(transition_matrix[current_point, :]) == 0:
                        break
                    source = np.random.choice(
                        range(transition_matrix.shape[0]),
                        p=transition_matrix[current_point, :]
                        / np.sum(transition_matrix[current_point, :]),
                    )
                    if (current_point, source) in edge_visit:
                        edge_visit[(current_point, source)] += 1
                    else:
                        edge_visit[(current_point, source)] = 1
                    current_point = source
                elif action == 2:
                    # Stay still
                    if (current_point, current_point) in edge_visit:
                        edge_visit[(current_point, current_point)] += 1
                    else:
                        edge_visit[(current_point, current_point)] = 1
                else:
                    pass
            # endregion
        if verbose:
            print("")

        # calculate new transition matrix from edge_visit
        new_transition_matrix = np.zeros_like(transition_matrix)
        for x_i in range(node_num):
            for y_i in range(node_num):
                if (x_i, y_i) in edge_visit:
                    new_transition_matrix[x_i, y_i] = edge_visit[(x_i, y_i)]
        new_transition_matrix = normalize_by_column(new_transition_matrix)
    else:
        new_transition_matrix = np.zeros_like(transition_matrix)
    # endregion
    # region BFS all possible paths
    if use_new_matrix:
        matrix = new_transition_matrix
    else:
        matrix = transition_matrix
    path_list = bfs(
        matrix,
        entry_point,
        reach_end=True,
        max_path_length=max_path_length,
        verbose=verbose,
    )
    # endregion
    # use different mean methods to estimate path joint probability
    path_list_prob = []
    for path in path_list:
        p = []
        end = path[0]
        for start in path[1:]:
            p.append(matrix[start, end])
            end = start
        if len(p) == 0:
            path_list_prob.append(0)
        else:
            if mean_method == "arithmetic":
                path_list_prob.append(np.mean(p))
            elif mean_method == "geometric":
                # Remove probability equal to 1 because they
                # don't contain useful information.
                p = [_ for _ in p if _ != 1]
                path_list_prob.append(stats.gmean(p))
            elif mean_method == "harmonic":
                # Remove probability equal to 1 because they
                # don't contain useful information.
                p = [_ for _ in p if _ != 1]
                path_list_prob.append(stats.hmean(p))

    # sort path by descending probability
    out = [item for item in zip(path_list_prob, path_list)]
    out.sort(key=lambda x: x[0], reverse=True)
    if verbose and verbose>=2:
        # verbose level >= 2: print backward paths in BFS
        if use_new_matrix:
            print(
                "{space:^15}{name:<30}: Father{action[0]:>8}, Child:{action[1]:>8}"
                ", Stay:{action[2]:>8}".format(
                    space="", name="Randwalk action", action=action_count
                )
            )
        for i in out[:10]:
            print("{:^0}{:<5.4f},{}".format("", i[0], [_ + 1 for _ in i[1]]))
    return out, new_transition_matrix
