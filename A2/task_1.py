import numpy as np

def normalize(array):
    return array*(1/np.sum(array))

# System
p0 = np.array([0.5, 0.5])
T = np.array([[0.8, 0.3], [0.2, 0.7]])
O_true = np.diag([0.75, 0.2])
O_false = np.diag([0.25, 0.8])
O = np.array([O_true, O_false])
evidence = np.array([1, 1, 0, 1, 0, 1])

def forward(e, current):
    return normalize(O[e] @ T.T @ current)

def predict(last_val, i):
    if i == evidence.shape[0]:
        return last_val
    res = normalize(T.T @ predict(last_val, i-1))
    print(f"P(X{(i)} | e1:{evidence.shape[0]}) = {res}")
    return res

def backward(e, current):
    return T @ O[e] @ current

def smooth(k):

    t = evidence.shape[0]
    current_b = np.ones(2)
    for i in range(t-k):
        current_b = backward(evidence[t-1-i], current_b)

    current_f = p0
    for i in range(k):
        current_f = forward(evidence[i], current_f)
    

    return normalize(current_f * current_b)


def forward_mle(e, current):
    return np.diag(O[e] * (T.T @ current).max(0))

if __name__ == '__main__':
    # b)
    current = p0
    for ind, e in enumerate(evidence):
        ind = ind + 1
        current = forward(e, current)
        print(f"P(X{(ind)} | e1:{ind}) = {current}")

    # c)
    predict(current, 30)

    # d)
    k = 1
    t = evidence.shape[0]
    for i in range(t):
        ind = i + 1
        print(f"P(X{(ind)} | e1:{t}) = {smooth(ind)}")

    # e)
    current = forward(evidence[0], p0)
    t = evidence.shape[0]
    path = np.zeros(t)
    for i, e in enumerate(evidence):
        ind = i + 1
        current = forward_mle(e, current)
        p = not np.argmax(current)
        path[i] = p
        print(f"argmax P(x1, x2, ..., X{(ind)} | e1:{ind}) = {path[0:ind]}")
