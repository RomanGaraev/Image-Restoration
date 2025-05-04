from reconstruct import DecompositionLayer

from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models import GoogLeNet_Weights, googlenet
from torchvision.models import AlexNet_Weights, alexnet
#from robustness.datasets import ImageNet
#from robustness import model_utils
from torch import nn


def load_regular_model(name="resnet50"):
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif name == "resnet101":
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    elif name == "googlenet":
        model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    elif name == "alexnet":
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    else:
        raise NotImplementedError(f"{name} is not recognised")
    model.eval()
    return model


def load_AT_model(resume_path="AT_models/imagenet_linf_8.pt"):
    pass
#    ds = ImageNet("imagenet val")
#    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
#                                                  resume_path=resume_path)
#    model.eval()
#    return model.model


class MyNet(nn.Module):
    def __init__(self, noise_percent=0.6):
        super(MyNet, self).__init__()
        self.decompose = DecompositionLayer(noise_percent=noise_percent, device='cpu')
        self.pretrained = load_regular_model()

    def forward(self, x):
        x = self.decompose(x)
        x = self.pretrained(x)
        return x


def load_robust_model(noise_percent=0.6):
    model = MyNet(noise_percent)
    model.eval()
    return model
