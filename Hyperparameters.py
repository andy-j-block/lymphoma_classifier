

class Hyperparameters:
    batch_size: int
    n_workers: int

    def __init__(self, batch_size: int, n_workers: int):
        self.batch_size = batch_size
        self.n_workers = n_workers
