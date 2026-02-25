from testing_datasets import ImageNetValidationDataset, CIFAR10Dataset
from utils import calculate_accuracy, compare_images, get_preferred_device
from models import load_regular_model_imagenet, load_regular_model_cifar, load_AT_model, load_robust_model
from reconstruct import DecompositionLayer

from collections import defaultdict
from tqdm import tqdm
import torchattacks
import numpy as np
import torch


def attack_test(attack,
                model_id="resnet50",
                visualize=False,
                method='svd',
                dataset_name="imagenet"):
    device = get_preferred_device()
    device = 'cpu'
    print(f"Device: {device}")

    dataset_name = dataset_name.lower()
    if dataset_name == "imagenet":
        dataset = ImageNetValidationDataset(root_dir='data/imagenet val (small)', model_id=model_id)
        model_loader = load_regular_model_imagenet
    elif dataset_name == "cifar10":
        dataset = CIFAR10Dataset(root_dir='data/cifar10', model_id=model_id, train=False, download=True)
        model_loader = load_regular_model_cifar
    else:
        raise ValueError(f"Unsupported dataset_name '{dataset_name}'. Use 'imagenet' or 'cifar10'.")

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=4,
                                             shuffle=True,
                                             num_workers=2,
                                             pin_memory=True)

    attack.set_normalization_used(mean=dataset.mean, std=dataset.std)
    # Untargeted attack
    attack.set_mode_default()

    total_accuracy = defaultdict(np.ndarray)
    for noise_percent in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # Model initialization
        decompose = DecompositionLayer(noise_percent=noise_percent,
                                       device=device, decomposition=method, rank=7, steps=30)
        model = model_loader(model_id).to(device)
        attack.set_model(model)
        attack.set_device(device)

        batch_accuracies = []
        for inputs, labels in (bar := tqdm(dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            adv_samples = attack(inputs, labels)
            restored_imgs = decompose(adv_samples.detach().clone()).to(device)

            if visualize:
                clean_pred = dataset.label(torch.argmax(model(inputs[0].unsqueeze(0))).cpu().item())
                adv_pred = dataset.label(torch.argmax(model(adv_samples[0].unsqueeze(0))).cpu().item())
                restored_pred = dataset.label(torch.argmax(model(restored_imgs[0].unsqueeze(0))).cpu().item())

                compare_images(dataset.inverse_transform(inputs[0].cpu().detach()).numpy(),
                               dataset.inverse_transform(adv_samples[0].cpu().detach()).numpy(),
                               dataset.inverse_transform(restored_imgs[0].cpu().detach()).numpy(),
                               clean_pred, adv_pred, restored_pred)
            #  
            outputs = model(restored_imgs)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = calculate_accuracy(labels, predicted)
            batch_accuracies.append(accuracy)
            bar.set_postfix({"Mean accuracy": np.mean(batch_accuracies)})
        total_accuracy[noise_percent] = np.mean(batch_accuracies)
        print(total_accuracy.items())
    return total_accuracy.items(), attack.__str__()


if __name__ == '__main__':
    dataset_name = "cifar10"
    n_classes = 1000 if dataset_name == "imagenet" else 10
    #atk = torchattacks.AutoAttack(torch.nn.Module(), eps=8/255, norm='Linf', n_classes=n_classes, version="plus")
    atk = torchattacks.PGD(torch.nn.Module(), eps=8/255 )
    model_ids = ["resnet50"]
    for model in model_ids:
        print(model)
        accuracies, attack_descr = attack_test(atk,
                                               model,
                                               visualize=False,
                                               method='svd',
                                               dataset_name=dataset_name)
        with open("logs.txt", 'w') as f:
            print(f"Attack: {attack_descr}.\n Accuracies: {accuracies}", file=f)
