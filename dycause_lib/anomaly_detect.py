"""The anomaly detection algorithm code.

Date: 2020-4-23
"""
import os

import matplotlib.pyplot as plt
import numpy as np


def anomaly_detect(
    data,
    weight=1,
    mean_interval=60,
    anomaly_proportion=0.3,
    verbose=True,
    save_fig=True,
    path_output=None,
):
    """Detect the time when anomaly first appears.

    Params:
        data: multi column data where each column represents a variable.
        weight: weight assigned to every variable when calculating anomaly
            scores.
        mean_interval: the size of sliding window to calculate standard deviation.
            Anomaly score within the first (mean_interval - 1) timestamps are 0.
        anomaly_proportion: proportion of anomaly variables considered to be
            anomaly, must relates to weight.
        verbose: the debugging print level: 0 (Nothing), 1 (Method info), 2 (Phase info), 3(Algorithm info)
    Returns:
        start_index: index in data when anomaly starts.
    """
    data_ma = []
    data_std = []

    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    for col in range(data.shape[1]):
        data_ma.append(
            np.concatenate(
                [
                    np.zeros([mean_interval - 1]),
                    moving_average(data[:, col], n=mean_interval),
                ],
                axis=0,
            )
        )
    data_ma = np.array(data_ma).T
    for col in range(data.shape[1]):
        one_std = [0 for i in range(mean_interval - 1)]
        for row in range(data.shape[0] - mean_interval):
            one_std.append(np.std(data[row : row + mean_interval, col]))
        data_std.append(one_std)
    data_std = np.array(data_std).T

    # Sum over nodes to get dither level of time
    # Here apply a weight of 1 to every node
    if weight == 1:
        dither_proportion = np.sum(
            (data_std > 1.0) * np.ones([data_std.shape[1]]), axis=1
        )

    start_time_list = [
        i
        for i in np.argsort(dither_proportion)[::-1]
        if dither_proportion[i] >= data.shape[1] * anomaly_proportion
    ]
    start_time = start_time_list[0]
    if verbose and verbose >= 3:
        # verbose level >= 3: print anomaly detection algorithm result
        print(
            "{space:^10}{name1:<30}: {}\n"
            "{space:^10}{name2:>30}: {}\n"
            "{space:^10}{name3:>30}:".format(
                start_time,
                dither_proportion[start_time],
                space="",
                name1="Max anomally start_time",
                name2="With score",
                name3="Others are",
            ),
            start_time_list[:10],
        )
    if save_fig:
        # Plot anomaly score of system
        fig = plt.figure(1)
        fig.set_size_inches(5, 3)
        fig.clear()
        fig.subplots_adjust(left=0.15, bottom=0.2)
        ax = plt.subplot(111)
        # pylint: disable=unsubscriptable-object
        plt.plot(np.arange(dither_proportion.shape[0]), dither_proportion, color='k')
        ax.axhline(y=anomaly_proportion*data.shape[1], color='r', linestyle='--', alpha=0.6)
        ax.text(x=0.4*len(data), y=anomaly_proportion*data.shape[1], s=r'$\theta * N$', 
                fontsize=15, horizontalalignment='right', verticalalignment='bottom')
        plt.xlim(0, data.shape[0])
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel("Score", fontsize=15)
        plt.savefig(
                    os.path.join(path_output, "anomaly-score-L{}.png".format(data.shape[0])), dpi=400
            )
        # Plot moving standard deviation & moving average for each service
        fig = plt.figure(1)
        nrows = int(np.sqrt(data.shape[1])) + 1
        fig.set_size_inches(nrows * 4, nrows * 3)
        fig.clear()
        axs = fig.subplots(nrows, nrows)
        fig.subplots_adjust(wspace=0.1, hspace=0.4)
        for i in range(data.shape[1]):
            ax = axs[i//nrows][i%nrows]
            ax.plot(range(data_ma.shape[0]), data_ma[:, i], color="b", alpha=0.4, label="MA")
            ax.plot(range(data_std.shape[0]), data_std[:, i], color="k", alpha=0.8, label="MSTD")
            ax.axhline(1.0, xmin=0, xmax=1, alpha=0.6, color="r", linestyle="--")
            ax.set_title('Service '+str(i + 1), fontdict={"fontsize": 15})
            ax.tick_params(axis='both', labelsize=15)
            ax.legend()
        plt.savefig(
            os.path.join(path_output, "mstd-all-L{}.png".format(data.shape[0]))
        )

    return start_time, dither_proportion[start_time]
