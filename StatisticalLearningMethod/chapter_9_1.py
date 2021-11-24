import numpy as np

y = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
pis = [0.46]
ps = [0.55]
qs = [0.67]
mus = np.zeros_like(y, dtype=float)
t = 0
while True:
    pi = pis[t]
    p = ps[t]
    q = qs[t]
    for i in range(len(mus)):
        a = pi * (p ** y[i]) * ((1 - p) ** (1 - y[i]))
        b = (1 - pi) * (q ** y[i]) * ((1 - q) ** (1 - y[i]))
        mus[i] = a / (a + b)
    pi_new = 1 / len(y) * np.sum(mus)
    p_new = np.sum(mus * y) / np.sum(mus)
    q_new = np.sum((1 - mus) * y) / np.sum(1 - mus)
    norm = np.linalg.norm([pi_new - pi, p_new - p, q_new - q])
    if norm <= 0.00005:
        break
    else:
        t += 1
        pis.append(pi_new)
        ps.append(p_new)
        qs.append(q_new)
print("pis", pis)
print("ps", ps)
print("qs", qs)
