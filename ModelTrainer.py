import optuna
from optuna.trial import TrialState
from PytorchAlgos import *
from KFolder import *
from typing import Union, Dict, Any
import time
import torch.optim as optim
from torch.optim import Adam, Adagrad
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


class ModelTrainer:
    train_dataloader: DataLoader
    train_dataset: Dataset
    valid_dataloader: DataLoader
    valid_dataset: Dataset
    valid_dataset_len: int
    test_dataloader: DataLoader
    test_dataset: Dataset

    algo_name: str
    algo: Union[ResNet, VGG, DenseNet]
    model: Union[ResNet, VGG, DenseNet]
    n_inner_fold: int

    hyperparams: Dict[str, Any]
    optimizer: Union[Adam, Adagrad]
    scheduler: Union[StepLR, ReduceLROnPlateau]

    best_trials: Dict[str, list]

    ###TODO integrate data into here correctly
    def __init__(self, pytorch_algos: PytorchAlgos, kfold_idxs: KFoldIndices, transformations: AlbTrxs):
        self.pytorch_algos = pytorch_algos
        self.kfold_idxs = kfold_idxs
        self.alb_trxs = transformations
        self.criterion = nn.CrossEntropyLoss()

    def set_dataloaders(self, n_outer_fold: int, n_inner_fold: int, batch_size: int, phases: Union[List[str], str]):
        if type(phases) is not list:
            phases = [phases]

        for phase in phases:
            if phase == 'train':
                df_train = DFTrainKFolded(n_outer_fold=n_outer_fold, n_inner_fold=n_inner_fold, kfold_idxs=self.kfold_idxs)
                df_train_dataloader_obj = DFTrainDataloader(kfolded_data=df_train, transformations=self.alb_trxs, batch_size=batch_size)
                self.train_dataloader, self.train_dataset = df_train_dataloader_obj.dataloader, df_train_dataloader_obj.dataset

            elif phase == 'valid':
                df_valid = DFValidKFolded(n_outer_fold=n_outer_fold, n_inner_fold=n_inner_fold, kfold_idxs=self.kfold_idxs)
                df_valid_dataloader_obj = DFValidDataloader(kfolded_data=df_valid, transformations=self.alb_trxs, batch_size=batch_size)
                self.valid_dataloader, self.valid_dataset = df_valid_dataloader_obj.dataloader, df_valid_dataloader_obj.dataset
                self.valid_dataset_len = len(df_valid_dataloader_obj.dataset)

            elif phase == 'test':
                df_test = DFTestKFolded(n_outer_fold=n_outer_fold, n_inner_fold=n_inner_fold, kfold_idxs=self.kfold_idxs)
                df_test_dataloader_obj = DFTestDataloader(kfolded_data=df_test, transformations=self.alb_trxs, batch_size=batch_size)
                self.test_dataloader, self.test_dataset = df_test_dataloader_obj.dataloader, df_test_dataloader_obj.dataset

    @staticmethod
    def create_hyperparam_grid(trial):
        return {
            'epochs': trial.suggest_int('epochs', 1, 100),
            'batch_size': trial.suggest_int('batch_size', 1, 16),
            'optimizer': trial.suggest_categorical('optimizer', [Adam, Adagrad]),
            'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
            'scheduler': trial.suggest_categorical('scheduler', [StepLR, ReduceLROnPlateau]),
            'alb_trxs_p': trial.suggest_float('alb_trxs_p', 0.1, 0.8),
            'alb_trxs_n_passes': trial.suggest_int('alb_trxs_n_passes', 0, 5),
            'step_size': trial.suggest_int('step_size', 5, 15)
        }

    def model_train(self, trial):
        self.hyperparams = self.create_hyperparam_grid(trial=trial)
        self.alb_trxs.p = self.hyperparams['alb_trxs_p']
        self.alb_trxs.n_passes = self.hyperparams['alb_trxs_n_passes']
        self.optimizer = getattr(optim, self.hyperparams['optimizer'])(self.model.parameters(), lr=self.hyperparams['lr'])
        self.scheduler = self.hyperparams['scheduler']

        accuracy: float = 0.0

        ###TODO define accuracy for function return
        for epoch in range(self.hyperparams['epochs']):
            train_loss = 0.0
            valid_correct = 0

            """train model"""
            self.model.train()
            for inputs, labels in self.train_dataloader:
                inputs = inputs.permute([0, 3, 2, 1])
                inputs, labels = inputs.to(self.model.torch_device), labels.to(self.model.torch_device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            self.scheduler.step()

            """validation eval"""
            self.model.eval()
            for inputs, labels in self.valid_dataloader:
                inputs = inputs.permute([0, 3, 2, 1])
                inputs, labels = inputs.to(self.model.torch_device), labels.to(self.model.torch_device)

                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                valid_correct += torch.sum(preds == labels.data)
            accuracy = valid_correct / self.valid_dataset_len

            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy

    def optuna_study(self, algo_name: str, n_inner_fold: int, show_param_importance: bool = True):
        """create a study on each of the inner fold complements to maximize the accuracy for given params/hyperparams.
           for each inner fold, store the best hyperparameters and save the model state dict learnable parameters"""

        start_time = time.time()
        study = optuna.create_study(direction='maximize')
        study.optimize(self.model_train, n_trials=20)
        elapsed_time = time.time() - start_time
        pruned_trials = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

        print('Study statistics:')
        print(f' Number of total trials: {len(study.trials)}')
        print(f' Number of pruned trials: {len(pruned_trials)}')
        print(f' Number of completed trials: {len(completed_trials)}')

        best_trial = study.best_trial
        param_importance = optuna.importance.get_param_importances(study)
        self.pytorch_algos.best_models[algo_name].append((n_inner_fold, best_trial.values, best_trial.params.items(), param_importance))
        torch.save(self.model.state_dict(), f'./Model_State_Dicts/{algo_name}/state_dict_{n_inner_fold}')

        print('Best trial:')
        print(f'  Accuracy: {best_trial.values}%')
        print(f'  Params:')
        for key, value in best_trial.params.items():
            print(f'    {key}: {value}')
        print(f'Time elapsed: {elapsed_time:.2f} sec')

        if show_param_importance:
            fig = optuna.visualization.plot_param_importances(study)
            fig.show()

    def tune_model(self):
        for algo_num, algo_name in enumerate(self.pytorch_algos.algos):  # RESNET18, RESNET 32, etc.
            for n_inner_fold in range(self.kfold_idxs.n_inner_splits):
                self.set_dataloaders(n_outer_fold=algo_num, n_inner_fold=n_inner_fold, batch_size=self.hyperparams['batch_size'], phases=['train', 'valid'])
                self.model = getattr(self.pytorch_algos, f'{algo_name}')
                self.optuna_study(algo_name=algo_name, n_inner_fold=n_inner_fold, show_param_importance=False)

    ### TODO lots of work on this
    def algo_eval(self):
        for algo_num, algo_name in enumerate(self.pytorch_algos.algos):
            best_model =

            valid_correct = 0
            self.model.eval()
            for inputs, labels in self.test_dataloader:
                inputs = inputs.permute([0, 3, 2, 1])
                inputs, labels = inputs.to(self.model.torch_device), labels.to(self.model.torch_device)

                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                valid_correct += torch.sum(preds == labels.data)
            accuracy = valid_correct / self.valid_dataset_len
