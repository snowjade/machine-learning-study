import numpy as np
import pandas as pd

features = ["年龄", "有工作", "有自己的房子", "信贷情况"]
X_train = pd.DataFrame([
    ["青年", "否", "否", "一般"],
    ["青年", "否", "否", "好"],
    ["青年", "是", "否", "好"],
    ["青年", "是", "是", "一般"],
    ["青年", "否", "否", "一般"],
    ["中年", "否", "否", "一般"],
    ["中年", "否", "否", "好"],
    ["中年", "是", "是", "好"],
    ["中年", "否", "是", "非常好"],
    ["中年", "否", "是", "非常好"],
    ["老年", "否", "是", "非常好"],
    ["老年", "否", "是", "好"],
    ["老年", "是", "否", "好"],
    ["老年", "是", "否", "非常好"],
    ["老年", "否", "否", "一般"]
], columns=['年龄', '有工作', '有自己的房子', '信贷情况'])
y_train = pd.DataFrame(["否", "否", "是", "是", "否", "否", "否", "是", "是", "是", "是", "是", "是", "是", "否"])


# data = X_train.copy()
# data.insert(data.shape[1], "类别", y_train)


class Node(object):
    def __init__(self):
        self.column = None
        self.parent = None
        self.children = {}
        self.kind = None


class C_45_TREE(object):

    def c_45_tree(self, X_train, y_train):
        node = Node()
        types = y_train[0].value_counts()
        node.kind = types.index[0]
        if len(types) == 1:
            return node
        print(type(y_train))
        entropy = self.entropy(y_train)
        best_gain = 0
        best_column = None
        for column in X_train.columns:
            if len(X_train[column].value_counts()) == 1:
                continue
            entropy_column = self.entropy_by_column(X_train, y_train, column)
            gain = (entropy - entropy_column) / self.entropy(X_train[column])
            if best_gain < gain:
                best_column = column
                best_gain = gain
        node.column = best_column
        node.gain = best_gain
        for value, _ in X_train[best_column].value_counts().items():
            select_index = X_train[best_column] == value
            node.children[value] = self.c_45_tree(X_train[select_index], y_train[select_index])
        return node

    # 计算column 的经验条件熵
    def entropy_by_column(self, X, y, column):
        total = len(y)
        value_counts = X[column].value_counts()
        if len(value_counts) == 1:
            return 0
        gain = 0
        for value, count in value_counts.items():
            entropy = self.entropy(y[X[column] == value])
            # entropy_column = self.entropy(X[column])
            gain += count / total * entropy
        return gain

    def entropy(self, y):
        if type(y) == pd.DataFrame:
            y = y[0]
        kinds = y.value_counts().to_numpy()
        kind_sum = np.sum(kinds)
        return -np.sum(np.log2(kinds / kind_sum) * kinds / kind_sum)


from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

label_x = preprocessing.LabelEncoder()
label_x.fit(np.unique(X_train))
x_unique = np.unique(X_train)
label_X_train = X_train.apply(label_x.transform)
label_y = preprocessing.LabelEncoder()
label_y.fit(np.unique(y_train))
label_y_train = y_train.apply(label_y.transform)

clf = DecisionTreeClassifier()
clf.fit(label_X_train, label_y_train)

from IPython.display import Image
from sklearn import tree
import pydotplus

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=features,
                                class_names=[str(k) for k in np.unique(y_train)],
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


def test():
    types = y_train[0].value_counts()
    for a, b in types.items():
        print("a", a)
    print(types)
    print(types.shape)
    print(len(types))
    print(types.index)
    print(types.sum())
    print(types.to_numpy())
    print(y_train[X_train["年龄"] == "青年"])


tree = C_45_TREE()
node = tree.c_45_tree(X_train, y_train)
print(node)
