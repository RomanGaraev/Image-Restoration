from reconstruct import DecompositionLayer

from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models import GoogLeNet_Weights, googlenet
from torchvision.models import AlexNet_Weights, alexnet
from torchvision.models import ConvNeXt_Large_Weights, convnext_large

from torch import nn

import timm

def _load_wideresnet70_16_from_timm():
    if timm is None:
        raise NotImplementedError(
            "WideResNet-70-16 requires timm. Install it with: pip install timm"
        )

    candidates = [
        "wide_resnet70_16",
        "wideresnet70_16",
        "wrn70_16",
    ]
    for model_name in candidates:
        try:
            return timm.create_model(model_name, pretrained=True, num_classes=1000)
        except Exception:
            continue

    raise NotImplementedError(
        "WideResNet-70-16 (ImageNet-1K pretrained) was not found in this timm version."
    )


def _load_swin_l_from_timm(name: str):
    if name == "swin_v2_l":
        candidates = [
            "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
            "swinv2_large_window12_192.ms_in22k_ft_in1k",
            "swinv2_large_window12_192",
            "swinv2_large_window12to16_192to256",
        ]
    else:
        candidates = [
            "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
            "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
            "swin_large_patch4_window7_224",
            "swin_large_patch4_window12_384",
        ]

    for model_name in candidates:
        try:
            return timm.create_model(model_name, pretrained=True, num_classes=1000)
        except Exception:
            continue

    raise NotImplementedError(
        f"{name} (ImageNet-1K pretrained) was not found in this timm version."
    )


def _load_meansparse_convnext_l_from_timm():
    candidates = [
        "meansparse_convnext_large",
        "mean_sparse_convnext_large",
        "meansparse_convnext_l",
        "mean_sparse_convnext_l",
        "convnext_large_meansparse",
    ]
    for model_name in candidates:
        try:
            return timm.create_model(model_name, pretrained=True, num_classes=1000)
        except Exception:
            continue

    raise NotImplementedError(
        "MeanSparse ConvNeXt-L (ImageNet-1K pretrained) was not found in this timm version."
    )


def _load_resnet50_cifar_from_timm():
    candidates = [
        "hf-hub:anonauthors/cifar10-timm-resnet50",
        "hf-hub:edadaltocg/resnet50_cifar10",
        "hf-hub:anonauthors/cifar10-resnet50",
    ]
    for model_name in candidates:
        try:
            return timm.create_model(model_name, pretrained=True)
        except Exception:
            continue

    raise NotImplementedError(
        "CIFAR-10 pretrained ResNet-50 checkpoint was not found in available timm/HF candidates."
    )


def load_regular_model_imagenet(name="resnet50"):
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif name == "resnet101":
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    elif name == "googlenet":
        model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    elif name == "alexnet":
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    elif name in ["swin_l", "swin_v2_l"]:
        model = _load_swin_l_from_timm(name)
    elif name in ["wideresnet70_16", "wide_resnet70_16", "wrn70_16"]:
        model = _load_wideresnet70_16_from_timm()
    elif name in ["meansparse_convnext_l", "mean_sparse_convnext_l", "meansparse_convnext_large"]:
        model = _load_meansparse_convnext_l_from_timm()
    elif name in ["convnext_l", "convnext_large"]:
        model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
    else:
        raise NotImplementedError(f"{name} is not recognised")
    model.eval()
    return model


def load_regular_model_cifar(name="resnet50"):
    if name != "resnet50":
        raise NotImplementedError(
            f"{name} is not supported for CIFAR yet. Supported: 'resnet50'."
        )
    model = _load_resnet50_cifar_from_timm()
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
        self.pretrained = load_regular_model_imagenet()

    def forward(self, x):
        x = self.decompose(x)
        x = self.pretrained(x)
        return x


def load_robust_model(noise_percent=0.6):
    model = MyNet(noise_percent)
    model.eval()
    return model
