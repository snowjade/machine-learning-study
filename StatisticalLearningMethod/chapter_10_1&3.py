import numpy as np

A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
pi = np.array([0.2, 0.4, 0.4])
O = np.array([0, 1, 0, 1])
T = O.size


def back(A, B, pi, O, T):
    beta = np.zeros((A.shape[0], T))
    beta[:, -1] = 1
    for i in range(T - 2, -1, -1):
        for j in range(0, A.shape[0]):
            beta[j, i] = np.sum(A[j, :] * beta[:, i + 1] * B[:, O[i + 1]])
    result = np.sum(pi * beta[:, 0] * B[:, O[0]])
    return result

print(back(A, B, pi, O, T))


def bestRoute(A, B, pi, O, T):
    theta = np.zeros((A.shape[0], T))
    psi = np.zeros((A.shape[0], T))
    theta[:, 0] = pi * B[:, O[0]]
    for i in range(1, T):
        theta[:, i] = np.max(theta[:, i - 1].reshape((-1, 1)) * A * B[:, O[i]], axis=0)
        psi[:, i] = np.argmax(theta[:, i - 1].reshape((-1, 1)) * A, axis=0)

    print("theta:", theta)
    print("psi:", psi)
    p = np.max(theta[:, T - 1])
    route = np.zeros(T, dtype=int)
    route[T - 1] = np.argmax(theta[:, T - 1])
    for i in range(T - 2, -1, -1):
        route[i] = psi[route[i + 1], i + 1]
    route = route + 1
    return p, route


print(bestRoute(A, B, pi, O, T))
