from collections import deque
import numpy as np


class Tracker:
    """R = rounds completed. last_selected[i] = 1-indexed round client i was last picked (0 = never).
    deadline[i] = last_selected[i] + T: client i must be picked in a round <= its deadline."""

    def __init__(self, n_clients, k, data_sizes, T):
        self.N, self.K, self.T = n_clients, k, T
        self.counts = np.zeros(n_clients, dtype=int)
        self.last_selected = np.zeros(n_clients, dtype=int)
        self.R = 0
        self.window = deque(maxlen=T)
        d = np.asarray(data_sizes, dtype=float)
        self.w = d / d.sum()

    def record(self, selected):
        self.R += 1
        for i in selected:
            self.counts[i] += 1
            self.last_selected[i] = self.R
        self.window.append(set(int(i) for i in selected))

    def deficit(self, mode="count"):
        expected = self.K * self.R / self.N if mode == "count" else self.w * self.K * self.R
        return expected - self.counts

    def window_count(self):
        c = np.zeros(self.N, dtype=int)
        for s in self.window:
            for i in s:
                c[i] += 1
        return c

    def deadlines(self):
        return self.last_selected + self.T