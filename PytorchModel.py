from abc import ABC, abstractmethod
from typing import Union
import torch
from torchvision.models import ResNet, VGG, DenseNet
import torch.nn as nn

# class Hyperparameters:
#     batch_size: int
#     n_workers: int
#
#     def __init__(self, model: Union[ResnetModel, VGGModel, DenseNetModel], batch_size: int, n_workers: int):
#         self.batch_size = batch_size
#         self.n_workers = n_workers
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#     scheduler = StepLR(optimizer, step_size=5, verbose=True)


class PytorchModel(ABC):
    def __init__(self, model: Union[ResNet, VGG, DenseNet], n_cancer_types=3):
        self.model = model
        self.n_cancer_types = n_cancer_types
        self.replace_fc_layers()
        # self.hyperparameters = self.set_hyperparameters()
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.send_to_device()

    def requires_grad(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @abstractmethod
    def replace_fc_layers(self):
        self.requires_grad()
        pass

    # def set_hyperparameters(self):
    #     return Hyperparameters(model=self.model, batch_size=4, n_workers=2)

    def send_to_device(self):
        self.model.to(self.torch_device)


class ResnetModel(PytorchModel, ResNet):
    def __init__(self, model: ResNet):
        super(ResNet, self).__init__()
        self.model = model

    def replace_fc_layers(self):
        self.requires_grad()
        self.model.fc = nn.Linear(self.model.fc.in_features, self.n_cancer_types)


class VGGModel(PytorchModel, VGG):
    def __init__(self, model: VGG):
        super(VGG, self).__init__()
        self.model = model

    def replace_fc_layers(self):
        self.requires_grad()
        self.model.classifier[6] = nn.Linear(4096, self.n_cancer_types)


class DenseNetModel(PytorchModel, DenseNet):
    def __init__(self, model: DenseNet):
        super(DenseNet, self).__init__()
        self.model = model

    def replace_fc_layers(self):
        self.requires_grad()
        self.model.classifier = nn.Linear(self.model.classifier.in_features, self.n_cancer_types)


# class Hyperparameters:
#     batch_size: int
#     n_workers: int
#
#     def __init__(self, model: Union[ResnetModel, VGGModel, DenseNetModel], batch_size: int, n_workers: int,
#                  criterion: nn = nn.CrossEntropyLoss()):
#         self.model = model
#         self.batch_size = batch_size
#         self.n_workers = n_workers
#
#         self.criterion = criterion
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
#         self.scheduler = StepLR(self.optimizer, step_size=5)
