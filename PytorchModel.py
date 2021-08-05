from abc import ABC, abstractmethod
import torchvision.models as models
import torch.nn as nn
import torch


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PytorchModel(ABC):
    def __init__(self, model, n_cancer_types=3):
        self.model = model
        self.n_cancer_types = n_cancer_types

    def requires_grad(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @abstractmethod
    def replace_fc_layers(self):
        self.requires_grad()
        pass

    def send_to_device(self, device):
        self.model.to(device)


class ResnetModel(PytorchModel, models):
    def replace_fc_layers(self):
        self.requires_grad()
        self.model.fc = nn.Linear(self.model.fc.in_features, self.n_cancer_types)


class VGGModel(PytorchModel, models):
    def replace_fc_layers(self):
        self.requires_grad()
        self.model.classifier[6] = nn.Linear(4096, self.n_cancer_types)


class DenseNetModel(PytorchModel, models):
    def replace_fc_layers(self):
        self.requires_grad()
        self.model.classifier = nn.Linear(self.model.classifier.in_features, self.n_cancer_types)
