import optuna
from optuna.trial import TrialState
from PytorchAlgos import *
from KFolder import *
from typing import Union, Dict, Any
import time
import torch.optim as optim
# from torch.optim import Adam, Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Torch Device: {TORCH_DEVICE}')

class ModelTrainer:
    train_dataloader: DataLoader
    train_dataset: Dataset
    valid_dataloader: DataLoader
    valid_dataset: Dataset
    valid_dataset_len: int
    test_dataloader: DataLoader
    test_dataset: Dataset
    test_dataset_len: int

    algo_name: str
    algo: Union[ResNet, VGG, DenseNet]
    model: Union[ResNet, VGG, DenseNet]

    n_outer_fold: int
    n_inner_fold: int

    hyperparams: Dict[str, Any]
    optimizer: Any #Union[Adam, Adagrad]
    scheduler: ReduceLROnPlateau
    batch_size: int

    ###TODO integrate data into here correctly
    def __init__(self, pytorch_algos: PytorchAlgos, kfold_idxs: KFoldIndices, transformations: AlbTrxs):
        self.pytorch_algos = pytorch_algos
        self.kfold_idxs = kfold_idxs
        self.alb_trxs = transformations
        self.criterion = nn.CrossEntropyLoss()

        """create multiindex, create results dataframe"""
        algos = self.pytorch_algos.algos
        n_inner_folds = range(self.kfold_idxs.n_inner_splits)
        index = pd.MultiIndex.from_product([algos, n_inner_folds], names=['algo', 'model_num'])
        self.results = pd.DataFrame(index=index, columns=['val_acc', 'test_acc', 'hyperparams', 'param_importance'])

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
                # self.valid_dataset_len = len(df_valid_dataloader_obj.dataset)

            elif phase == 'test':
                df_test = DFTestKFolded(n_outer_fold=n_outer_fold, n_inner_fold=n_inner_fold, kfold_idxs=self.kfold_idxs)
                df_test_dataloader_obj = DFTestDataloader(kfolded_data=df_test, transformations=self.alb_trxs, batch_size=batch_size)
                self.test_dataloader, self.test_dataset = df_test_dataloader_obj.dataloader, df_test_dataloader_obj.dataset
                # self.test_dataset_len = len(df_test_dataloader_obj.dataset)

    @staticmethod
    def create_hyperparam_grid(trial):
        return {
            'epochs': trial.suggest_int('epochs', 1, 100),
            'batch_size': trial.suggest_int('batch_size', 1, 32),
            'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'Adagrad']),
            'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
            'alb_trxs_p': trial.suggest_float('alb_trxs_p', 0.1, 0.8),
            'alb_trxs_n_passes': trial.suggest_int('alb_trxs_n_passes', 0, 5),
        }

    def model_train(self, trial):
        self.hyperparams = self.create_hyperparam_grid(trial=trial)
        self.alb_trxs.p = self.hyperparams['alb_trxs_p']
        self.alb_trxs.n_passes = self.hyperparams['alb_trxs_n_passes']
        self.optimizer = getattr(optim, str(self.hyperparams['optimizer']))(self.model.parameters(), lr=self.hyperparams['lr'])
        self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer)
        self.batch_size = self.hyperparams['batch_size']

        self.set_dataloaders(n_outer_fold=self.n_outer_fold, n_inner_fold=self.n_inner_fold, batch_size=self.batch_size, phases=['train', 'valid'])

        accuracy: float = 0.0

        for epoch in range(self.hyperparams['epochs']):#self.epochs):
            print(f'Epoch {epoch}')
            train_start = time.time()
            train_loss = 0.0
            valid_correct = 0.0

            """train model"""
            self.model.train()
            for inputs, labels in self.train_dataloader:
                inputs, labels = inputs.to(TORCH_DEVICE), labels.to(TORCH_DEVICE)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            """validation eval"""
            self.model.eval()
            for inputs, labels in self.valid_dataloader:
                inputs, labels = inputs.to(TORCH_DEVICE), labels.to(TORCH_DEVICE)

                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                valid_correct += torch.sum(preds == labels.data)
            accuracy = valid_correct / self.valid_dataset.length
            self.scheduler.step(accuracy)
            print(f'Accuracy: {accuracy}')
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            print(f'Train time: {time.time() - train_start}')

        return accuracy

    def optuna_study(self, algo_name: str, n_outer_fold: int, n_inner_fold: int, show_param_importance: bool = True):
        """create a study on each of the inner fold complements to maximize the accuracy for given params/hyperparams.
           for each inner fold, store the best hyperparameters and save the model state dict learnable parameters"""
        self.n_outer_fold, self.n_inner_fold = n_outer_fold, n_inner_fold

        start_time = time.time()
        study = optuna.create_study(direction='maximize')
        study.optimize(self.model_train, n_trials=3, timeout=150)
        elapsed_time = time.time() - start_time
        pruned_trials = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

        print('Study statistics:')
        print(f' Number of total trials: {len(study.trials)}')
        print(f' Number of pruned trials: {len(pruned_trials)}')
        print(f' Number of completed trials: {len(completed_trials)}')

        best_trial = study.best_trial
        try:
            param_importance = optuna.importance.get_param_importances(study)
        except RuntimeError:
            param_importance = np.nan

        """store results, save learnable parameters, test accuracy (nan) to be filled in later"""
        self.results.loc[algo_name, :] = (best_trial.values, np.nan, best_trial.params.items(), param_importance)
        torch.save(self.model.state_dict(), f'./Model_State_Dicts/{algo_name}/state_dict_{n_inner_fold}.pth')

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
        for algo_num, algo_name in enumerate(self.pytorch_algos.algos):
            for n_inner_fold in range(self.kfold_idxs.n_inner_splits):
                self.model = getattr(self.pytorch_algos, f'{algo_name}').model
                self.optuna_study(algo_name=algo_name, n_outer_fold=algo_num, n_inner_fold=n_inner_fold, show_param_importance=False)

    def model_selection(self):
        """sort the results dataframe by validation accuracy, create tuples of the multiindex indices to drop, drop them"""
        self.results = self.results.sort_values(by='val_acc', ascending=False)
        drop_rows_list = [self.results.loc[algo].index[1:] for algo in self.pytorch_algos.algos]
        drop_rows_tuples = [[(self.pytorch_algos.algos[i], drop_rows_list[i][j]) for j, _ in enumerate(drop_rows_list[i])] for i, _ in enumerate(drop_rows_list)]
        drop_rows_tuples = [tuples for sublist in drop_rows_tuples for tuples in sublist]
        self.results = self.results.drop(drop_rows_tuples)
        self.results = self.results.reset_index(level=1)

    ### TODO lots of work on this
    def model_scoring(self):
        for algo_num, algo_name in enumerate(self.pytorch_algos.algos):
            self.model = getattr(self.pytorch_algos, algo_name)
            self.model.load_state_dict(torch.load(f'./Model_State_Dicts/{algo_name}/state_dict_{self.results["model_num"].loc[algo_name]}.pth'))

            test_correct = 0
            self.model.eval()
            for inputs, labels in self.test_dataloader:
                inputs = inputs.permute([0, 3, 2, 1])
                inputs, labels = inputs.to(self.model.torch_device), labels.to(self.model.torch_device)

                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                test_correct += torch.sum(preds == labels.data)
            accuracy = test_correct / self.test_dataset_len

            return accuracy

