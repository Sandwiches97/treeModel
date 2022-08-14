import sys

import xgboost as xgb



if __name__ == "__main__":

    print(-float("inf")*2)
    print()
    a = "lksdadsf ajfwe"
    b = list(a)
    c = set(b)
    c = list(c)
    c.sort()
    import ctypes

    lis = [0, 1,2,4]
    lis2 = [0, 1,2,4]
    print(c, id(0), ctypes.cast(id(8), ctypes.py_object).value)

    # xgb_train = xgb.DMatrix('data/agaricus.txt.train')
    # xgb_test = xgb.DMatrix('data/agaricus.txt.test')
    # params = {
    #     "objective": "binary:logistic",
    #     "booster": "gbtree",
    #     "max_depth": 3
    # }
    # num_round = 5
    # watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
    # model = xgb.train(params, xgb_train, num_round, watchlist)