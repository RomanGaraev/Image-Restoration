#import kagglehub

# Download latest version
#path = kagglehub.dataset_download("kerrit/imagenet1kmediumtest-10k")

#print("Path to dataset files:", path)

import torch
from pathlib import Path

from robustbench import benchmark
from robustbench.data import PREPROCESSINGS, load_imagenet
from models import MyNet

threat_model = "Linf"  # one of {"Linf", "L2", "corruptions"}
dataset = "imagenet"  # one of {"cifar10", "cifar100", "imagenet"}
#data_dir = Path("/Users/romangaraev/.cache/kagglehub/datasets/kerrit/imagenet1kmediumtest-10k/versions/1/test_10000/test_10000/") 
preprocessing = PREPROCESSINGS["Res256Crop224"]

model = MyNet()
model_name = "Ahmadi2026TT_Reconstruct"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Validate dataset loading with RobustBench's ImageNet loader to ensure compatibility.
x_ref, y_ref = load_imagenet(n_examples=16, transforms_test=preprocessing)
print(f"RobustBench load_imagenet OK: {x_ref.shape=}, {y_ref.shape=}")

# For local ImageNet models that are not in RobustBench model zoo, preprocessing must be specified.
clean_acc, robust_acc = benchmark(model, model_name=model_name, n_examples=5000, dataset=dataset,
                                  threat_model=threat_model, eps=8/255, device=device,
                                  to_disk=True, preprocessing=preprocessing,
                                  data_dir=str(data_dir))
print(clean_acc)
print(robust_acc)
