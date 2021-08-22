import optuna
from PytorchModel import *
from KFolder import *
from typing import Union
import time
import copy
from torch.optim import Adam, Adagrad
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchvision.models as models



class ModelTrainer:
    ###TODO integrate data into here correctly
    def __init__(self, model: Union[ResnetModel, VGGModel, DenseNetModel],
                 kfold_idxs: KFoldIndices,
                 transformations: AlbumentationsTransformations, hyperparameters: Hyperparameters):
        self.model = model
        self.kfold_idxs = kfold_idxs
        self.hyperparameters = hyperparameters
        self.albumentation_transformations = transformations

    def create_dataloaders(self, n_outer_fold: int, n_inner_fold: int, phase: str):
        if phase == 'train':
            df_train = DFTrainKFolded(n_outer_fold=n_outer_fold,
                                      n_inner_fold=n_inner_fold,
                                      kfold_idxs=self.kfold_idxs)
            df_train_dataloader_obj = DFTrainDataloader(kfolded_data=df_train,
                                                        transformations=self.albumentation_transformations,
                                                        hyperparameters=self.hyperparameters)

            df_train_dataloader = df_train_dataloader_obj.dataloader
            df_train_dataset = df_train_dataloader_obj.dataset

            return df_train_dataloader, df_train_dataset

        elif phase == 'valid':
            df_valid = DFValidKFolded(n_outer_fold=n_outer_fold,
                                      n_inner_fold=n_inner_fold,
                                      kfold_idxs=self.kfold_idxs)
            df_valid_dataloader_obj = DFValidDataloader(kfolded_data=df_valid,
                                                        transformations=self.albumentation_transformations,
                                                        hyperparameters=self.hyperparameters).dataloader

            df_valid_dataloader = df_valid_dataloader_obj.dataloader
            df_valid_dataset = df_valid_dataloader_obj.dataset

            return df_valid_dataloader, df_valid_dataset

    def training_set_eval(self, phase='train'):
        self.model.train()

        for outer_fold in range(self.kfold_idxs.n_outer_splits):
            for inner_fold in range(self.kfold_idxs.n_inner_splits):
                training_dataloader, training_dataset = self.create_dataloaders(n_outer_fold=outer_fold,
                                                                                n_inner_fold=inner_fold,
                                                                                phase=phase)

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in training_dataloader:
                    print(labels)
                    inputs = inputs.permute([0, 3, 2, 1])
                    inputs = inputs.to(self.model.torch_device)
                    labels = labels.to(self.model.torch_device)

                    # zero out the gradients before training
                    self.optimizer.zero_grad()

                    # set gradient calculations ON for training
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.model.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        loss.backward()
                        self.model.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                self.scheduler.step()

                epoch_loss = running_loss / len(dataloader[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)

    def validation_set_eval(self, phase='valid'):
        self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        for outer_fold in range(self.kfold_idxs.n_outer_splits):
            for inner_fold in range(self.kfold_idxs.n_inner_splits):
                valid_dataloader, valid_dataset = self.create_dataloaders(n_outer_fold=outer_fold,
                                                                          n_inner_fold=inner_fold,
                                                                          phase=phase)

                for inputs, labels in valid_dataloader:
                    print(labels)
                    inputs = inputs.permute([0, 3, 2, 1])
                    inputs = inputs.to(self.model.torch_device)
                    labels = labels.to(self.model.torch_device)

                    # set gradient calculations ON for training
                    with torch.set_grad_enabled(False):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloader[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)

                print(f'{phase} loss: {epoch_loss:.3f}, accuracy: {epoch_acc:.3f}')

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                val_acc_history.append(epoch_acc.item())
                val_loss_history.append(epoch_loss)
                lr_history.append(optimizer.param_groups[0]['lr'])

    def objective(self, trial):
        params = {
            'batch_size': trial.suggest_int('batch_size', 1, 16),
            'optimizer': trial.suggest_categorical('optimizer', [Adam, Adagrad]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-2, 1e-5),
            'scheduler': trial.suggest_categorical('scheduler', [StepLR, ReduceLROnPlateau]),
            'step_size': trial.suggest_int('step_size', 5, 15)
        }


    def run_training_optuna(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=20)

        best_trial = study.best_trial
        print(best_trial.values)
        print(best_trial.params)






###############

    def training_steps(self, dataloader, model, criterion, optimizer, scheduler, epochs=25, **inputs):
        start = time.time()

        val_acc_history = []
        val_loss_history = []
        lr_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(epochs):
            print(f'Epoch {epoch +1}/{epochs}')
            print('- ' * 15)

            self.training_set_eval()
            self.validation_set_eval()

            print()

        time_elapsed = time.time() - start
        print(f'Training time: {time_elapsed // 60}m {time_elapsed % 60}s')
        print()

        model.load_state_dict(best_model_wts)

        return model, val_acc_history, val_loss_history, lr_history