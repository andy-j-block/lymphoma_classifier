from PytorchModel import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import optuna

class Hyperparameters:
    batch_size: int
    n_workers: int

    def __init__(self, model: Union[ResnetModel, VGGModel, DenseNetModel], batch_size: int, n_workers: int):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=5)

    def objective(self, trial):
        params = {
            'batch_size': trial.suggest_int('batch_size', 1, 16),
            'optimizer': trial.suggest_categorical('optimizer', [optim.Adam, optim.Adagrad]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-2, 1e-5),
            'scheduler': trial.suggest_categorical('scheduler', [StepLR, ReduceLROnPlateau]),
            'step_size': trial.suggest_int('step_size', 5, 15)
        }