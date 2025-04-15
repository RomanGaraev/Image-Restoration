from reconstruct import DecompositionLayer

from torchvision.models import ResNet50_Weights, resnet50
from robustness.datasets import ImageNet
from robustness import model_utils
from torch import nn


def load_regular_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    return model


def load_AT_model(resume_path="AT_models/imagenet_linf_8.pt"):
    ds = ImageNet("imagenet val")
    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                  resume_path=resume_path)
    model.eval()
    return model.model


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
