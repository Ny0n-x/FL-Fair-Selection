class Selector:
    """client_losses = dict {client_id: loss} for ONLY the clients whose loss was queried.
    Random / RoundRobin / our rules: {}. Power-of-Choice: its d candidates. DivFL: all N.
    len(client_losses) is the 'query cost' we report."""
    def __init__(self, n_clients, k, data_sizes, cfg):
        self.N, self.K, self.cfg, self.data_sizes = n_clients, k, cfg, data_sizes

    def candidates(self, round_idx):
        """Which clients the loop should query for loss BEFORE select(). Default: none."""
        return []

    def select(self, round_idx, client_losses):
        raise NotImplementedError

    def update(self, round_idx, selected, client_losses):
        pass