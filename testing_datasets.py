from torchvision.models import ResNet50_Weights, ResNet101_Weights, GoogLeNet_Weights, AlexNet_Weights
from typing import Callable, Dict, Iterable, List, Optional
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.ops import box_iou
from torchvision import transforms
from PIL import Image
import torch
import json
import os


class ImageNetValidationDataset(Dataset):
    def __init__(self, root_dir='imagenet val', model_id="resnet50"):
        super(ImageNetValidationDataset, self).__init__()

        # Load labels from JSON
        with open(os.path.join(root_dir, "Labels.json"), 'r') as f:
            self.labels_dict = json.load(f)
        with open(os.path.join(root_dir, "imagenet_extended.txt")) as f:
            lines = f.readlines()
            self.imagenet_labels = {int(line.split(":")[0]): line.split(":")[1] for line in lines}
        # Create mapping from synset IDs to class indices
        self.syn_to_class = {}
        for syn_id, x in self.labels_dict.items():
            class_id = -1
            for i, class_names in self.imagenet_labels.items():
                if x in class_names:
                    class_id = i
            if class_id == -1:
                print(x)
            self.syn_to_class[syn_id] = class_id

        # Get list of all image files
        self.image_files = []
        for syn_id in self.syn_to_class.keys():
            syn_folder = os.path.join(os.path.join(root_dir, "val"), syn_id)
            if os.path.exists(syn_folder):
                self.image_files.extend([os.path.join(syn_folder, img)
                                         for img in os.listdir(syn_folder)])

        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]

        self.model_id = model_id
        if model_id == "resnet50":
            self.transforms = ResNet50_Weights.IMAGENET1K_V2.transforms()
        elif model_id == "resnet101":
            self.transforms = ResNet101_Weights.IMAGENET1K_V2.transforms()
        elif model_id == "alexnet":
            self.transforms = AlexNet_Weights.IMAGENET1K_V1.transforms()
        elif model_id == "googlenet":
            self.transforms = GoogLeNet_Weights.IMAGENET1K_V1.transforms()
        else:
            print(f"{model_id} is not defined. Use default transformations")
            self.transforms = transforms.Compose([transforms.Resize((232, 232)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=self._mean,
                                                                       std=self._std)])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        syn_id = os.path.basename(os.path.dirname(img_path))
        label = self.syn_to_class[syn_id]
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        return image, label

    def inverse_transform(self, x):
        # Thanks to Saurabh https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / i for i in self._std]),
                                       transforms.Normalize(mean=[-i for i in self._mean],
                                                            std=[1., 1., 1.]),
                                       ])
        return invTrans(x)

    def label(self, ind):
        return self.imagenet_labels[ind].split(",")[0].replace("'", "")

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std


class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir='data/cifar10', model_id="resnet50", train=False, download=True):
        super(CIFAR10Dataset, self).__init__()
        self.root_dir = root_dir
        self.train = train

        # Keep normalization compatible with ImageNet-pretrained classifiers by default.
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]

        if model_id == "resnet50":
            self.transforms = ResNet50_Weights.IMAGENET1K_V2.transforms()
        elif model_id == "resnet101":
            self.transforms = ResNet101_Weights.IMAGENET1K_V2.transforms()
        elif model_id == "alexnet":
            self.transforms = AlexNet_Weights.IMAGENET1K_V1.transforms()
        elif model_id == "googlenet":
            self.transforms = GoogLeNet_Weights.IMAGENET1K_V1.transforms()
        else:
            self.transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=self._mean,
                                                                       std=self._std)])

        self.dataset = CIFAR10(root=self.root_dir,
                               train=self.train,
                               transform=self.transforms,
                               download=download)
        self.class_names = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def inverse_transform(self, x):
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / i for i in self._std]),
                                       transforms.Normalize(mean=[-i for i in self._mean],
                                                            std=[1., 1., 1.]),
                                       ])
        return invTrans(x)

    def label(self, ind):
        return self.class_names[ind]

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std
