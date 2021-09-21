import os
import optuna
from optuna.trial import TrialState
from ImageLoader import label_decoder
from PytorchAlgos import *
from KFolder import *
from typing import Union, Dict, Any
import time
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Torch device: {TORCH_DEVICE}')


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
    algo: Union[ResNet, VGG]
    model: Union[ResNet, VGG]

    n_outer_fold: int
    n_inner_fold: int

    hyperparams: Dict[str, Any]
    optimizer: Any #Union[Adam, Adagrad]
    scheduler: ReduceLROnPlateau
    batch_size: int

    y: np.ndarray
    y_pred: np.ndarray

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
        self.results = pd.DataFrame(index=index, columns=['val_acc', 'test_acc', 'hyperparams', 'param_importance',
                                                          'train_set_len', 'valid_set_len', 'test_set_len', 'n_outer_fold'])

    def set_dataloaders(self, n_outer_fold: int, phases: Union[List[str], str], n_inner_fold: int = None, batch_size: int = None):
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

            elif phase == 'test':
                df_test = DFTestKFolded(n_outer_fold=n_outer_fold, kfold_idxs=self.kfold_idxs)
                df_test_dataloader_obj = DFTestDataloader(kfolded_data=df_test, transformations=self.alb_trxs, batch_size=len(df_test.nkf_df))
                self.test_dataloader, self.test_dataset = df_test_dataloader_obj.dataloader, df_test_dataloader_obj.dataset

    @staticmethod
    def create_hyperparam_grid(trial):
        return {
            'epochs': trial.suggest_int('epochs', 10, 50),
            'batch_size': trial.suggest_int('batch_size', 1, 32),
            'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'Adagrad']),
            'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
            'alb_trxs_p': trial.suggest_float('alb_trxs_p', 0.1, 0.5),
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

        for epoch in range(self.hyperparams['epochs']):
            print(f'Epoch {epoch+1}')
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
            print(f'Train loss: {train_loss}')

            """validation eval"""
            self.model.eval()
            for inputs, labels in self.valid_dataloader:
                inputs, labels = inputs.to(TORCH_DEVICE), labels.to(TORCH_DEVICE)

                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                valid_correct += torch.sum(preds == labels.data)
            accuracy = valid_correct / len(self.valid_dataset)
            self.scheduler.step(accuracy)
            print(f'Accuracy: {accuracy}')
            trial.report(accuracy, epoch)
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()

            print(f'Train time: {time.time() - train_start}')

        return accuracy

    def optuna_study(self, algo_name: str, n_outer_fold: int, n_inner_fold: int, show_param_importance: bool = True):
        """create a study on each of the inner fold complements to maximize the accuracy for given params/hyperparams.
           for each inner fold, store the best hyperparameters and save the model state dict learnable parameters"""
        self.n_outer_fold, self.n_inner_fold = n_outer_fold, n_inner_fold

        start_time = time.time()
        study = optuna.create_study(direction='maximize')
        print(f'Begin training {algo_name} model')
        study.optimize(self.model_train, n_trials=8)
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
        except (ValueError, RuntimeError):
            param_importance = np.nan

        """store results, save learnable parameters, nans to be filled in later"""
        self.results.loc[algo_name, :] = (best_trial.values, np.nan, best_trial.params.items(), param_importance,
                                          self.train_dataset.length, self.valid_dataset.length, np.nan, self.n_outer_fold)
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

        study_df = study.trials_dataframe()
        study_df.to_csv(f'./Model_State_Dicts/{algo_name}/state_dict_{n_inner_fold}.csv')

    def tune_model(self):
        for algo_num, algo_name in enumerate(self.pytorch_algos.algos):
            for n_inner_fold in range(self.kfold_idxs.n_inner_splits):
                self.model = getattr(self.pytorch_algos, f'{algo_name}').model
                self.optuna_study(algo_name=algo_name, n_outer_fold=algo_num, n_inner_fold=n_inner_fold, show_param_importance=False)

    def model_selection(self):
        # if 'tuned_results.csv' in os.listdir(os.getcwd()):
        #     self.results = pd.read_csv('selected_results.csv', index_col=('algo', 'model_num'))

        """sort the results dataframe by validation accuracy, create tuples of the multiindex indices to drop, drop them"""
        self.results = self.results.sort_values(by='val_acc', ascending=False)
        drop_rows_list = [self.results.loc[algo].index[1:] for algo in self.pytorch_algos.algos]
        drop_rows_tuples = [[(self.pytorch_algos.algos[i], drop_rows_list[i][j]) for j, _ in enumerate(drop_rows_list[i])] for i, _ in enumerate(drop_rows_list)]
        drop_rows_tuples = [tuples for sublist in drop_rows_tuples for tuples in sublist]
        self.results = self.results.drop(drop_rows_tuples).reset_index(level=1)

    def model_scoring(self):
        # if 'selected_results.csv' in os.listdir(os.getcwd()):
        #     self.results = pd.read_csv('selected_results.csv', index_col='algo')

        for algo_num, algo_name in enumerate(self.pytorch_algos.algos):
            self.model = getattr(self.pytorch_algos, algo_name).model

            self.model.load_state_dict(torch.load(f'./Model_State_Dicts/{algo_name}/state_dict_{self.results["model_num"].loc[algo_name]}.pth'))

            self.set_dataloaders(phases='test', n_outer_fold=algo_num)

            test_correct: float = 0.0
            self.model.eval()
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(TORCH_DEVICE), labels.to(TORCH_DEVICE)

                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                test_correct += torch.sum(preds == labels.data)
            accuracy = test_correct / len(self.test_dataset)
            print(f'Accuracy: {accuracy}')
            self.results['test_acc'].loc[algo_name] = float(accuracy)
            self.results['test_set_len'].loc[algo_name] = self.test_dataset.length

    def get_predictions(self):
        """first get the best model, run the best model again and save predictions"""
        # if 'scored_results.csv' in os.listdir(os.getcwd()):
        #     self.results = pd.read_csv('scored_results.csv', index_col='algo')

        self.results = self.results.reset_index()
        self.results = self.results.sort_values(by='test_acc', ascending=False).iloc[3]

        self.model = self.pytorch_algos.RESNET101.model
        self.model.load_state_dict(torch.load(f'./Model_State_Dicts/RESNET101/state_dict_0.pth')) #{self.results["n_outer_fold"]}.pth')
        # self.model = getattr(self.pytorch_algos, self.results['algo']).model
        # self.model.load_state_dict(torch.load(f'./Model_State_Dicts/{self.results["algo"]}/state_dict_0.pth')) #{self.results["n_outer_fold"]}.pth'))

        self.set_dataloaders(phases='test', n_outer_fold=0)
        self.model.eval()
        for inputs, labels in self.test_dataloader:
            inputs, labels = inputs.to(TORCH_DEVICE), labels.to(TORCH_DEVICE)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

            self.y, self.y_pred = labels.cpu().numpy(), preds.cpu().numpy()

        data = {'y': self.y, 'y_pred': self.y_pred}
        self.results = pd.DataFrame(data=data)

        for col in self.results.columns:
            self.results[col] = self.results[col].map(label_decoder())

        self.save_results('final_results.csv')

    def save_results(self, filename: str):
        self.results.to_csv(filename)

