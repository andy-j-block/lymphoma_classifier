import optuna
from optuna.trial import TrialState
from PytorchModel import *
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

    hyperparams: Dict[str, Any]
    optimizer: Union[Adam, Adagrad]
    scheduler: Union[StepLR, ReduceLROnPlateau]
    param_importance: Dict[str, float]

    ###TODO integrate data into here correctly
    def __init__(self, model: Union[ResNetModel, VGGModel, DenseNetModel],
                 kfold_idxs: KFoldIndices,
                 transformations: AlbTrxs):
        self.model = model
        self.kfold_idxs = kfold_idxs
        self.alb_trxs = transformations
        self.criterion = nn.CrossEntropyLoss()

    # ### TODO use setattr or getattr? gotta be a way to make this code cleaner
    # def create_dataloaders(self, n_outer_fold: int, n_inner_fold: int, batch_size: int, phase: str):
    #     if phase == 'train':
    #         df_train = DFTrainKFolded(n_outer_fold=n_outer_fold, n_inner_fold=n_inner_fold, kfold_idxs=self.kfold_idxs)
    #         df_train_dataloader_obj = DFTrainDataloader(kfolded_data=df_train, transformations=self.alb_trxs, batch_size=batch_size)
    #
    #         df_train_dataloader, df_train_dataset = df_train_dataloader_obj.dataloader, df_train_dataloader_obj.dataset
    #         return df_train_dataloader, df_train_dataset
    #
    #     elif phase == 'valid':
    #         df_valid = DFValidKFolded(n_outer_fold=n_outer_fold, n_inner_fold=n_inner_fold, kfold_idxs=self.kfold_idxs)
    #         df_valid_dataloader_obj = DFValidDataloader(kfolded_data=df_valid, transformations=self.alb_trxs, batch_size=batch_size)
    #
    #         df_valid_dataloader, df_valid_dataset = df_valid_dataloader_obj.dataloader, df_valid_dataloader_obj.dataset
    #         return df_valid_dataloader, df_valid_dataset
    #
    #     elif phase == 'test':
    #         df_test = DFTestKFolded(n_outer_fold=n_outer_fold, n_inner_fold=n_inner_fold, kfold_idxs=self.kfold_idxs)
    #         df_test_dataloader_obj = DFTestDataloader(kfolded_data=df_test, transformations=self.alb_trxs, batch_size=batch_size)
    #
    #         df_test_dataloader, df_test_dataset = df_test_dataloader_obj.dataloader, df_test_dataloader_obj.dataset
    #         return df_test_dataloader, df_test_dataset

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


        ###TODO define accuracy for function return
        ###TODO move the data portion out of the training loop for objective maximization
        for epoch in range(self.hyperparams['epochs']):
            start_time = time.time()
            train_loss = 0.0
            valid_correct = 0.0

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
                # running_corrects += torch.sum(preds == labels.data)
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
            accuracy = valid_correct / len(self.valid_dataset)

            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            end_time = time.time() - start_time
        return accuracy

    def optuna_optimizer(self, show_param_importance: bool = True):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.model_train, n_trials=20)

        pruned_trials = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

        print('Study statistics:')
        print(f' Number of total trials: {len(study.trials)}')
        print(f' Number of pruned trials: {len(pruned_trials)}')
        print(f' Number of completed trials: {len(completed_trials)}')

        best_trial = study.best_trial
        print('Best trial:')
        print(f'  Accuracy: {best_trial.values}%')
        print(f'  Params:')
        for key, value in best_trial.params.items():
            print(f'    {key}: {value}')

        self.param_importance = optuna.importance.get_param_importances(study)
        if show_param_importance:
            fig = optuna.visualization.plot_param_importances(study)
            fig.show()

    def tune_model(self):

        for inner_fold in range(self.kfold_idxs.n_inner_splits):
            df_train = DFTrainKFolded(n_outer_fold=n_outer_fold, n_inner_fold=inner_fold, kfold_idxs=self.kfold_idxs)
            self.train_dataloader = DFTrainDataloader(kfolded_data=df_train, transformations=self.alb_trxs, batch_size=self.hyperparams['batch_size']).dataloader
            self.train_dataset = DFTrainDataloader(kfolded_data=df_train, transformations=self.alb_trxs, batch_size=self.hyperparams['batch_size']).dataset
            df_valid = DFValidKFolded(n_outer_fold=n_outer_fold, n_inner_fold=inner_fold, kfold_idxs=self.kfold_idxs)
            self.valid_dataloader = DFValidDataloader(kfolded_data=df_valid, transformations=self.alb_trxs, batch_size=self.hyperparams['batch_size']).dataloader
            self.valid_dataset = DFValidDataloader(kfolded_data=df_valid, transformations=self.alb_trxs, batch_size=self.hyperparams['batch_size']).dataset


    def algo_eval(self):
        for outer_fold in range(self.kfold_idxs.n_outer_splits):
            for inner_fold in range(self.kfold_idxs.n_inner_splits):
                """generate the augmented train data and valid data"""
                train_dataloader, train_dataset = \
                    self.create_dataloaders(n_outer_fold=outer_fold, n_inner_fold=inner_fold, batch_size=self.hyperparams['batch_size'], phase='train')
                valid_dataloader, valid_dataset = \
                    self.create_dataloaders(n_outer_fold=outer_fold, n_inner_fold=inner_fold, batch_size=self.hyperparams['batch_size'], phase='valid')


    #
    # def validation_set_eval(self, phase='valid'):
    #     self.model.eval()
    #
    #     running_loss = 0.0
    #     running_corrects = 0
    #     for outer_fold in range(self.kfold_idxs.n_outer_splits):
    #         for inner_fold in range(self.kfold_idxs.n_inner_splits):
    #             valid_dataloader, valid_dataset = self.create_dataloaders(n_outer_fold=outer_fold,
    #                                                                       n_inner_fold=inner_fold,
    #                                                                       phase=phase)
    #
    #             for inputs, labels in valid_dataloader:
    #                 print(labels)
    #                 inputs = inputs.permute([0, 3, 2, 1])
    #                 inputs = inputs.to(self.model.torch_device)
    #                 labels = labels.to(self.model.torch_device)
    #
    #                 # set gradient calculations ON for training
    #                 with torch.set_grad_enabled(False):
    #                     outputs = self.model(inputs)
    #                     loss = self.criterion(outputs, labels)
    #                     _, preds = torch.max(outputs, 1)
    #
    #                 running_loss += loss.item() * inputs.size(0)
    #                 running_corrects += torch.sum(preds == labels.data)
    #
    #             epoch_loss = running_loss / len(dataloader[phase].dataset)
    #             epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)
    #
    #             print(f'{phase} loss: {epoch_loss:.3f}, accuracy: {epoch_acc:.3f}')
    #
    #             if epoch_acc > best_acc:
    #                 best_acc = epoch_acc
    #                 best_model_wts = copy.deepcopy(model.state_dict())
    #
    #             val_acc_history.append(epoch_acc.item())
    #             val_loss_history.append(epoch_loss)
    #             lr_history.append(optimizer.param_groups[0]['lr'])
    #

###############

    # def training_steps(self, dataloader, model, criterion, optimizer, scheduler, epochs=25, **inputs):
    #     start = time.time()
    #
    #     val_acc_history = []
    #     val_loss_history = []
    #     lr_history = []
    #
    #     best_model_wts = copy.deepcopy(model.state_dict())
    #     best_acc = 0.0
    #
    #     for epoch in range(epochs):
    #         print(f'Epoch {epoch +1}/{epochs}')
    #         print('- ' * 15)
    #
    #         self.training_set_eval()
    #         self.validation_set_eval()
    #
    #         print()
    #
    #     time_elapsed = time.time() - start
    #     print(f'Training time: {time_elapsed // 60}m {time_elapsed % 60}s')
    #     print()
    #
    #     model.load_state_dict(best_model_wts)
    #
    #     return model, val_acc_history, val_loss_history, lr_history