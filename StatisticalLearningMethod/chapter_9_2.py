import numpy as np
import scipy.stats as st

y = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75])
model_num = 2
gamma = np.zeros((y.size, model_num), dtype=float)
alpha = [0.5, 0.5]
miu = [1, 0]
theta = [10, 10]

while True:
    alpha_ori = alpha.copy()
    miu_ori = miu.copy()
    theta_ori = theta.copy()
    for j in range(y.size):
        pdf_j = []
        for k in range(model_num):
            pdf_j.append(alpha[k] * st.norm.pdf(y[j], loc=miu[k], scale=theta[k]))

        pdf_sum_j = np.sum(pdf_j)
        for k in range(model_num):
            gamma[j, k] = pdf_j[k] / pdf_sum_j
    for k in range(model_num):
        theta[k] = np.sqrt(np.sum((y - miu[k]) ** 2 * gamma[:, k]) / np.sum(gamma[:, k]))
        alpha[k] = np.sum(gamma[:, k]) / y.size
        miu[k] = np.sum(y * gamma[:, k]) / (np.sum(gamma[:, k]))

    print("miu", miu)
    print("theta", theta[0] ** 2, theta[1] ** 2)
    print("alpha", alpha)
    if np.linalg.norm(np.array(alpha + miu + theta) - np.array(alpha_ori + miu_ori + theta_ori)) < 0.5:
        break
