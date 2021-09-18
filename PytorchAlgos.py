from abc import ABC, abstractmethod
from typing import Union, Dict, List, Tuple, Any
import torch
import torchvision.models as models
from torchvision.models import ResNet, VGG, DenseNet
import torch.nn as nn


class ModelAbstract(ABC):
    n_cancer_types: int = 3
    torch_device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def __init__(self, model: Union[ResNet, VGG, DenseNet]):
        self.model = model

    def requires_grad(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @abstractmethod
    def replace_fc_layers(self):
        self.requires_grad()
        pass

    def send_to_device(self):
        self.model.to(self.torch_device)


class CustomResNet(ModelAbstract, ResNet):
    def __init__(self, model: ResNet):
        super(ResNet, self).__init__()
        self.model = model
        self.replace_fc_layers()
        self.send_to_device()

    def replace_fc_layers(self):
        self.requires_grad()
        self.model.fc = nn.Linear(self.model.fc.in_features, self.n_cancer_types)


class CustomVGG(ModelAbstract, VGG):
    def __init__(self, model: VGG):
        super(VGG, self).__init__()
        self.model = model
        self.send_to_device()

    def replace_fc_layers(self):
        self.requires_grad()
        self.model.classifier[6] = nn.Linear(4096, self.n_cancer_types)


class CustomDenseNet(ModelAbstract, DenseNet):
    def __init__(self, model: DenseNet):
        super(DenseNet, self).__init__()
        self.model = model
        self.send_to_device()

    def replace_fc_layers(self):
        self.requires_grad()
        self.model.classifier = nn.Linear(self.model.classifier.in_features, self.n_cancer_types)


class PytorchAlgos:

    RESNET18: ResNet = CustomResNet(models.resnet18(pretrained=True))
    RESNET34: ResNet = CustomResNet(models.resnet34(pretrained=True))
    RESNET101: ResNet = CustomResNet(models.resnet101(pretrained=True))

    ### TODO uncomment these later
    VGG13: VGG = CustomVGG(models.vgg13(pretrained=True))
    VGG16: VGG = CustomVGG(models.vgg16(pretrained=True))
    VGG13_BN: VGG = CustomVGG(models.vgg13_bn(pretrained=True))
    VGG16_BN: VGG = CustomVGG(models.vgg16_bn(pretrained=True))

    DENSENET121: DenseNet = CustomDenseNet(models.densenet121(pretrained=True))
    DENSENET169: DenseNet = CustomDenseNet(models.densenet169(pretrained=True))
    DENSENET201: DenseNet = CustomDenseNet(models.densenet201(pretrained=True))

    def __init__(self):
        self.algos = [algo for algo in dir(self) if '__' not in algo]
        self.n_algos: int = len(self.algos)
