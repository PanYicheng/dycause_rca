import math
import time

import numpy as np


def calc_pearson(matrix, method='default', zero_diag=True):
    """Calculate the pearson correlation between nodes

    Params:
        matrix: data of shape [N, T], N is node num, T is sample num
        method: method used, default for manually calculation,
            numpy for numpy implementation
        zero_diag:
                if zero the self correlation value (in diagonal position)
    """
    # 一行是一个数据
    if method == 'numpy':
        res = np.corrcoef(np.array(matrix))
        if zero_diag:
            for i in range(res.shape[0]):
                res[i, i] = 0.0
        res = res.tolist()
    else:
        nrows = len(matrix)
        ncols = len(matrix[0])
        n = ncols * 1.0
        res = [[0 for i in range(nrows)] for j in range(nrows)]
        for i in range(nrows):
            idx = i + 1
            for j in range(idx, nrows):
                a = b = c = f = e = 0
                for k in range(0, ncols):
                    a += matrix[i][k] * matrix[j][k]  # sigma xy
                    b += matrix[i][k]  # sigma x
                    c += matrix[j][k]  # sigma y
                    e += matrix[i][k] * matrix[i][k]  # sigma xx
                    f += matrix[j][k] * matrix[j][k]  # sigma yy

                para1 = a
                para2 = b * c / n
                para3 = e
                para4 = b * b / n
                para5 = f
                para6 = c * c / n

                r1 = para1 - para2
                r2 = (para3 - para4) * (para5 - para6)
                r2 = math.sqrt(r2)
                r = 1.0 * r1 / r2
                res[i][j] = res[j][i] = r * 1.00000
        # w = Workbook()
        # # Write to a new Sheet
        # ws = w.create_sheet("relation"+str(ela) + ".xlsx")  #Create table sheet
        # for i in range(nrows):
        #     for j in range(nrows):
        # 	    ws.cell(row = i+1, column = j+1).value = res[i][j]
        if not zero_diag:
            for i in range(nrows):
                for j in range(nrows):
                    res[i][j] = 1.0
    return res


if __name__ == "__main__":
    np.random.seed(42)
    a = np.random.rand(100, 100)
    tic = time.time()
    b_numpy = calc_pearson(a, 'numpy')
    time_numpy = time.time() - tic
    tic = time.time()
    b_default = calc_pearson(a, 'default')
    time_default = time.time() - tic
    print('{:^10}Default time:{}'.format('', time_default))
    print('{:^10}Numpy   time:{}'.format('', time_numpy))
    print('{:^10}Distance    :{}'.format(
        '', np.linalg.norm(np.array(b_numpy)-np.array(b_default))))
    print('Result numpy  :', b_numpy[0][:10])
    print('Result default:', b_default[0][:10])
