import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_alldata(data, data_head, filepath):
    fig = plt.figure(1)
    nrows = int(np.sqrt(data.shape[1])) + 1
    fig.set_size_inches(10 * nrows, 8 * nrows)
    fig.clear()
    axs = fig.subplots(nrows, nrows)
    for i in range(data.shape[1]):
        ax = axs[i//nrows][i%nrows]
        ax.plot(range(data.shape[0]), data[:, i], color='k', alpha=0.8)
        ax.set_title(str(i + 1) + ":" + data_head[i], fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
    plt.savefig(filepath, dpi=300)


def draw_overlay_histogram(histogram, title, filepath):
    fig = plt.figure(1)
    fig.set_size_inches(5, 3)
    fig.clear()
    # pylint: disable=unsubscriptable-object
    plt.plot(range(np.array(histogram).shape[0]), histogram)
    plt.title(title + "(Sum:{})".format(sum(histogram)))
    plt.savefig(filepath, dpi=400)


def draw_bar_histogram(histogram, auto_threshold_ratio,  title, filepath):
    fig = plt.figure(1)
    fig.set_size_inches(5, 3)
    fig.clear()
    ax = fig.subplots(1, 1)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.2, wspace=0.2, hspace=0.35)
    # pylint: disable=unsubscriptable-object
    ax.bar(range(1, np.array(histogram).shape[0] + 1), histogram, color="k")
    ax.set_xticks(range(1, len(histogram)+1))
    ax.tick_params(axis='x', labelsize=9, labelrotation=60)
    ax.set_xlabel('Services', fontsize='large', horizontalalignment='center')
    ax.axhline(y=auto_threshold_ratio * np.max(histogram), color="r", alpha=0.8, 
                linestyle="--", label=r'$\theta_e * N$')
    ax.legend()
    plt.title(title, fontsize='large')
    plt.savefig(filepath, dpi=400)
