import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
train_y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])
train_X = pd.DataFrame(train_X)
train_y = pd.DataFrame(train_y)


class Node(object):
    def __init__(self):
        self.feature = None


class CartTree(object):

    def __init__(self):
        self.root_node = None

    def fit(self, X, y):
        self.root_node = self.rec_fit(X, y)

    def predict(self, X):
        y = []
        for i in range(len(X)):
            y.append(self.rec_predict(X.iloc[i], self.root_node))
        return np.array(y)

    def rec_predict(self, x, node):
        if node.feature is None:
            return node.predict_y
        if x[node.feature] <= node.value:
            return self.rec_predict(x, node.left)
        else:
            return self.rec_predict(x, node.right)

    def rec_fit(self, X, y):
        node = Node()
        if len(X) == 1:
            node.predict_y = y[0]
            return node
        if self.cal_loss(y)[0] < 0.1:
            node.predict_y = y.mean()
            return node

        best_feature = None
        min_loss = None
        best_uni = None
        for feature in X.columns:
            unis = np.unique(X[feature])
            for uni in unis:
                index_left = X[feature] <= uni
                index_right = X[feature] >= uni
                loss = self.cal_loss(y[index_left])[0] + self.cal_loss(y[index_right])[0]
                if min_loss is None or min_loss > loss:
                    min_loss = loss
                    best_feature = feature
                    best_uni = uni
        node.feature = best_feature
        node.value = best_uni
        node.left = self.rec_fit(X[X[best_feature] <= best_uni], y[X[best_feature] <= best_uni])
        node.right = self.rec_fit(X[X[best_feature] > best_uni], y[X[best_feature] > best_uni])
        node.predict_y = y.mean()
        return node

    def cal_loss(self, y):
        if y.empty:
            return 0
        c = y.mean()
        return 1 / len(y) * ((y - c) ** 2).sum()


cartTree = CartTree()
cartTree.fit(train_X, train_y)
test_X = pd.DataFrame(np.linspace(0, 10, 1000))
test_y = cartTree.predict(test_X)
a, b = test_X.iloc[:, 0], test_y[:, 0]
plt.plot(test_X.iloc[:, 0], test_y[:, 0])
plt.show()
