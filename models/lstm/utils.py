import numpy as np

# Fenêtrage (N, 18) → (N//win, win, 18)
def _make_windows(X, y, win):
    X_seq, y_seq = [], []
    for i in range(0, len(X) - win + 1, win):      # stride = win
        X_seq.append(X[i : i + win])
        y_seq.append(y[i + win - 1])               # étiquette du dernier pas
    return np.stack(X_seq), np.array(y_seq)