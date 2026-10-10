def get_partition(dataset, alpha, n_clients, seed):
    """Return (train_idx, test_idx, data_sizes): per-client lists, 80/20 split.
    Must be identical for the same seed."""
    raise NotImplementedError