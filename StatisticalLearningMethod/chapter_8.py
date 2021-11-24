import numpy as np

# 加载训练数据
X = np.array([[0, 1, 3],
              [0, 3, 1],
              [1, 2, 2],
              [1, 1, 3],
              [1, 2, 3],
              [0, 1, 2],
              [1, 1, 2],
              [1, 1, 1],
              [1, 3, 1],
              [0, 2, 1]
              ])
y = np.array([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1])


class Stump:
    def __init__(self, column, value, y_left, y_right):
        self.column = column
        self.value = value
        self.y_left = y_left
        self.y_right = y_right


class AdaBoostTree:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.trees = []
        self.stumps = []
        self.alphas = []

    def fit(self):
        weight = np.full(shape=y.size, fill_value=1 / y.size)
        while True:
            e_best = None
            print("e_best", e_best)
            stump = None
            types_y = np.unique(y)
            result_best = None
            for column in range(X.shape[1]):
                for value in np.unique(X[:, column]):
                    y_predict = y.copy()
                    y_predict[X[:, column] <= value] = types_y[0]
                    y_predict[X[:, column] > value] = types_y[1]
                    result = np.zeros_like(y)
                    result[y_predict != y] = 1
                    e = np.sum(weight * result)
                    print(result)
                    print("e", e)
                    if e_best is None or e < e_best:
                        e_best = e
                        stump = Stump(column, value, types_y[0], types_y[1])
                        result_best = result

                    y_predict[X[:, column] <= value] = types_y[1]
                    y_predict[X[:, column] > value] = types_y[0]
                    result = np.zeros_like(y)
                    result[y_predict != y] = 1
                    e = np.sum(weight * result)
                    if e_best is None or e < e_best:
                        e_best = e
                        stump = Stump(column, value, types_y[1], types_y[0])
                        result_best = result
            self.stumps.append(stump)
            print("e_best", e_best)
            alpha = 1 / 2 * np.log((1 - e_best) / e_best)
            self.alphas.append(alpha)
            result_best[result_best == 0] = -1
            weight = weight * np.exp(alpha * result_best)
            weight = weight / np.sum(weight)
            if self.cal_error(y, self.predict(X)) <= 0:
                break

    @staticmethod
    def cal_error(y, y_predict):
        return np.sum(y != y_predict) / y.size

    def predict(self, X):
        results = []
        for x in X:
            result = 0
            for i in range(len(self.stumps)):
                stump = self.stumps[i]
                result += self.alphas[i] * (stump.y_left if x[stump.column] <= stump.value else stump.y_right)
            results.append(-1 if result <= 0 else 1)
        return np.array(results)


ada_boost_tree = AdaBoostTree(X, y)
ada_boost_tree.fit()
print("y_original", y)
print("y_predict", ada_boost_tree.predict(X))
