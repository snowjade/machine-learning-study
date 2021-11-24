import numpy as np

A = np.array([[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]])
B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
pi = np.array([0.2, 0.3, 0.5])
O = np.array([0, 1, 0, 0, 1, 0, 1, 1])
T = O.size


def back(A, B, pi, O, T):
    beta = np.zeros((A.shape[0], T))
    beta[:, -1] = 1
    for i in range(T - 2, -1, -1):
        for j in range(0, A.shape[0]):
            beta[j, i] = np.sum(A[j, :] * beta[:, i + 1] * B[:, O[i + 1]])
    result = np.sum(pi * beta[:, 0] * B[:, O[0]])
    return result, beta


def forward(A, B, pi, O, T):
    beta = np.zeros((A.shape[0], T))
    beta[:, 0] = pi * B[:, O[0]]
    for i in range(1, T):
        for j in range(0, A.shape[0]):
            beta[j, i] = np.sum(beta[:, i - 1] * A[:, j] * B[j, O[i]])
    result = np.sum(beta[:, T - 1])
    return result, beta


back_result, back_beta = back(A, B, pi, O, T)
forward_result, forward_beta = forward(A, B, pi, O, T)
print(back_result)
print(forward_result)
print(back_beta)
# P(i4=q3|O,lamda)
p = (forward_beta[2, 3] * back_beta[2, 3]) / forward_result
print(p)
