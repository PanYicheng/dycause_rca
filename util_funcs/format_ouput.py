"""Tools for formatting metric to use in excel and latex
"""
import numpy as np


def format_to_excel(prkS, my_acc):
    ret = ""
    for prk in prkS:
        ret += " {:.4f}".format(prk)
    ret += " {:.4f}".format(np.mean(prkS))
    ret += " {:.4f}".format(my_acc)
    return ret


def format_to_latex(prkS, my_acc):
    ret = ""
    for prk in prkS:
        ret += " {:.2f} &".format(prk * 100.0)
    ret += " {:.2f}".format(np.mean(prkS) * 100.0)
    ret += " {:.2f} \\\\".format(my_acc * 100.0)
    return ret


if __name__ == "__main__":
    print(format_to_excel([0, 0.1, 0.2], 0.75))
    print(format_to_latex([0, 0.1, 0.2], 0.75))
