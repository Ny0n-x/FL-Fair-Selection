class Selector:
    """ctx provides:
      ctx.loss_fn(ids)   -> losses of the CURRENT global model on those clients
      ctx.update_fn(ids) -> local update vectors (trains those clients)
      ctx.updates        -> {client_id: update} for clients trained this round (read in update())
    Selectors count their own extra client queries in self.extra_queries."""
    def __init__(self, n_clients, k, data_sizes, cfg):
        self.N, self.K, self.cfg, self.data_sizes = n_clients, k, cfg, data_sizes
        self.extra_queries = 0

    def select(self, round_idx, ctx):
        raise NotImplementedError

    def update(self, round_idx, selected, ctx):
        pass