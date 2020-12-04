

import numpy as np
# 统计每个变量的缺失值占比


def count_na(data):
    cols = data.columns.tolist()    # cols为data的所有列名
    n_df = data.shape[0]    # n_df为数据的行数
    for col in cols:
        missing = np.count_nonzero(data[col].isnull().values)  # col列中存在的缺失值个数
        mis_perc = float(missing) / n_df * 100
        print("{col}的缺失比例是{miss}%".format(col=col, miss=mis_perc))
