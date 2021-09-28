from abc import ABC, abstractmethod
from typing import Union
import torch
import torchvision.models as models
from torchvision.models import ResNet, VGG, MobileNetV2
import torch.nn as nn


class ModelAbstract(ABC):
    n_cancer_types: int = 3
    torch_device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def __init__(self, model: Union[ResNet, VGG, MobileNetV2]):
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


class CustomMobileNet(ModelAbstract, MobileNetV2):
    def __init__(self, model: MobileNetV2):
        super(MobileNetV2, self).__init__()
        self.model = model
        self.send_to_device()

    def replace_fc_layers(self):
        self.requires_grad()
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.n_cancer_types, bias=True)


class PytorchAlgos:
    """IMPORTANT NOTE: half of these algorithms should be commented out before training
       so that the test data set isn't too small"""

    def __init__(self, resnet18=True, resnet34=True, resnet101=True,
                 vgg13=True, vgg16=True, vgg13_bn=True, vgg16_bn=True, mobilenet_v2=True):

        self.resnet18__, self.resnet34__, self.resnet101__, self.vgg13__, self.vgg16__, self.vgg13_bn__, self.vgg16_bn__, \
            self.mobilenet_v2__ = resnet18, resnet34, resnet101, vgg13, vgg16, vgg13_bn, vgg16_bn, mobilenet_v2

        if self.resnet18__:
            self.RESNET18 = CustomResNet(models.resnet18(pretrained=True))

        if self.resnet34__:
            self.RESNET34 = CustomResNet(models.resnet34(pretrained=True))

        if self.resnet101__:
            self.RESNET101 = CustomResNet(models.resnet101(pretrained=True))

        if self.vgg13__:
            self.VGG13 = CustomVGG(models.vgg13(pretrained=True))

        if self.vgg16__:
            self.VGG16 = CustomVGG(models.vgg16(pretrained=True))

        if self.vgg13_bn__:
            self.VGG13_BN = CustomVGG(models.vgg13_bn(pretrained=True))

        if self.vgg16_bn__:
            self.VGG16_BN = CustomVGG(models.vgg16_bn(pretrained=True))

        if self.mobilenet_v2__:
            self.MOBILENET_V2 = CustomMobileNet(models.mobilenet_v2(pretrained=True))

        self.algos = [algo for algo in dir(self) if '__' not in algo]
        self.n_algos: int = len(self.algos)
