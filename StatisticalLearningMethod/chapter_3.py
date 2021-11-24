from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

np.random.seed(544)
x = np.random.random((40, 2))
y = np.random.randint(0, 2, 40)
# x = np.array([[1, 1], [3, 3], [2, 2], [4, 4], [5, 5], [6, 6], [7, 7]])
# y = np.array([0, 1, 1, 0, 1, 0, 1])

plt.figure(figsize=(15, 15))


class Node(object):
    def __init__(self, data):
        self.data = data
        self.parent = None
        self.left_child = None
        self.right_child = None


class KDTree:
    def __init__(self):
        self.root = None
        self.visited_nodes = set()
        self.neighbours = []

    ##就是构造 kd 树
    def fit(self, x, y):
        data = np.hstack((x, y.reshape(x.shape[0], 1)))
        self.root = self.get_tree(None, data, 0)

    def get_tree(self, parent, data, row):
        data = data[data[:, row % (data.shape[1] - 1)].argsort()]
        middle = int(data.shape[0] / 2)
        node = Node(data[middle, :])
        node.parent = parent

        if middle != 0:
            node.left_child = self.get_tree(node, data[:middle, :], row + 1)
        if middle != data.shape[0] - 1:
            node.right_child = self.get_tree(node, data[middle + 1:, :], row + 1)
        return node

    # 找到以node为根节点的树的x点的叶子节点
    def find_point(self, x, node, row):
        if node.right_child is None and node.left_child is None:
            return node, row
        elif node.left_child is None:
            print("find_point node.right_child", node.right_child)
            return self.find_point(x, node.right_child, row + 1)
        elif node.right_child is None:
            return self.find_point(x, node.left_child, row + 1)
        row = row % (node.data.size - 1)
        if x[row] >= node.data[row]:
            return self.find_point(x, node.right_child, row + 1)
        elif x[row] < node.data[row]:
            return self.find_point(x, node.left_child, row + 1)

    @staticmethod
    def distance(x, node):
        return np.linalg.norm(x - node.data[:-1])

    def maximum_distance(self, x):
        neighbours_point = np.array([node.data for node in self.neighbours])[:, :-1]
        return max(np.linalg.norm(neighbours_point - x, ord=2, axis=1))

    def find_neighbours(self, x, k):
        self.neighbours.clear()
        self.visited_nodes.clear()
        self.handle_sub_tree(x, self.root, 0, k)
        neighbours_point = np.array([node.data for node in self.neighbours])
        return neighbours_point

    def predict(self, x, k):
        y = []
        for point in x:
            self.find_neighbours(point, k)
            neighbours = self.find_neighbours(point, k)
            y.append(Counter(neighbours[:, -1]).most_common(1)[0][0])
        return np.array(y)

    # 处理某个节点，主要是判断是否需要加入到k邻域
    def handle_node(self, x, node, k):
        self.visited_nodes.add(node)
        if len(self.neighbours) < k:
            self.neighbours.append(node)
        else:
            node_distance = self.distance(x, node)
            neighbours_point = np.array([node.data for node in self.neighbours])[:, :-1]
            distances = np.linalg.norm(neighbours_point - x, ord=2, axis=1)
            if max(distances) > node_distance:
                self.neighbours[distances.argmax()] = node

    def handle_sub_tree(self, x, node, row, k):
        node, row = self.find_point(x, node, row)
        self.handle_node(x, node, k)
        self.handle_parent(x, node, row, k)

    def handle_parent(self, x, node, row, k):
        if node.parent is None:
            return
        if node.parent in self.visited_nodes:
            self.handle_parent(x, node.parent, row - 1, k)
            return
        parent_node = node.parent
        self.handle_node(x, parent_node, k)
        other_node = parent_node.left_child if parent_node.right_child is node else parent_node.right_child
        if other_node is not None and (
                len(self.neighbours) < k or abs(x[row - 1] - parent_node.data[row - 1]) < self.maximum_distance(x)):
            self.handle_sub_tree(x, other_node, row, k)
        else:
            self.handle_parent(x, parent_node, row - 1, k)


def test():
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit(x, y)
    print("x shape", x.shape)
    point = [0.3, 0.7]
    distances, neighbors = model.kneighbors([point], 5)
    neighbors = x[neighbors[0]]
    neighbors = neighbors[neighbors[:, 0].argsort()]
    print("sklearn neighbors", neighbors)

    kd_tree = KDTree()
    kd_tree.fit(x, y)
    kd_neighbors = kd_tree.find_neighbours(point, 5)
    kd_neighbors = kd_neighbors[kd_neighbors[:, 0].argsort()]
    print("kd neighbors", kd_neighbors)


test()


def draw(k):
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(x, y)
    plt.subplot(3, 3, k)
    n = 256
    l = np.linspace(0, 1, n)
    x0, x1 = np.meshgrid(l, l)
    y_predict = model.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
    cmap = ListedColormap(("green", "red"))
    plt.contourf(x0, x1, y_predict, cmap=cmap, alpha=0.5)
    plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.5, cmap=cmap, s=50, edgecolors='k')


#
# for i in tqdm(range(1, 10)):
#     draw(i)
# plt.show()


def draw_kd(k):
    model = KDTree()
    model.fit(x, y)
    plt.subplot(3, 3, k)
    n = 256
    l = np.linspace(0, 1, n)
    x0, x1 = np.meshgrid(l, l)
    y_predict = model.predict(np.c_[x0.ravel(), x1.ravel()], k).reshape(x0.shape)
    cmap = ListedColormap(("green", "red"))
    plt.contourf(x0, x1, y_predict, cmap=cmap, alpha=0.5)
    plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.5, cmap=cmap, s=50, edgecolors='k')


for i in tqdm(range(1, 10)):
    draw(i)
plt.show()
