import numpy as np
from tabulate import tabulate


def prCal(scoreList, prk, rightOne):
    """计算scoreList的prk值

    Params:
        scoreList: list of tuple (node, score)
        prk: the top n nodes to consider
        rightOne: ground truth nodes
    """
    prkSum = 0
    for k in range(min(prk, len(scoreList))):
        if scoreList[k][0] in rightOne:
            prkSum = prkSum + 1
    denominator = min(len(rightOne), prk)
    return prkSum / denominator


def pr_stat(scoreList, rightOne, k=5):
    topk_list = range(1, k + 1)
    prkS = [0] * len(topk_list)
    for j, k in enumerate(topk_list):
        prkS[j] += prCal(scoreList, k, rightOne)
    return prkS


def print_prk_acc(prkS, acc):
    headers=['PR@{}'.format(i+1) for i in range(len(prkS))]+['PR@Avg', 'Acc']
    data = prkS + [np.mean(prkS)]
    data.append(acc)
    print(tabulate([data], headers=headers, floatfmt="#06.4f"))


def my_acc(scoreList, rightOne, n=None):
    """Accuracy for Root Cause Analysis with multiple causes.
    Refined from the Acc metric in TBAC paper.
    """
    node_rank = [_[0] for _ in scoreList]
    if n is None:
        n = len(scoreList)
    s = 0.0
    for i in range(len(rightOne)):
        if rightOne[i] in node_rank:
            rank = node_rank.index(rightOne[i]) + 1
            s += (n - max(0, rank - len(rightOne))) / n
        else:
            s += 0
    s /= len(rightOne)
    return s
