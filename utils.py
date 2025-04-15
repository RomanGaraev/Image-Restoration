from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import torch


def calculate_accuracy(labels, predictions):
    correct = 0
    for pred, label in zip(predictions, labels):
        if pred.item() == label.item():
            correct += 1
    return 100.0 * correct / labels.size(0)


def compare_images(clean_img, adv_image, restored_img, clean_pred: str, adv_pred: str, restored_pred:str):
    print(f"PSNR between clean and adversarial samples: {peak_signal_noise_ratio(clean_img, adv_image)}")
    print(f"PSNR between clean and restored samples: {peak_signal_noise_ratio(clean_img, restored_img)}")

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    assert clean_img.shape == restored_img.shape
    for ind, (name, img) in enumerate(zip(["Original" + f" ({clean_pred})",
                                           "Adversarial" + f" ({adv_pred})",
                                           "Restored" + f" ({restored_pred})"],
                                          [clean_img, adv_image, restored_img])):
        ax[ind].imshow(img.transpose(1, 2, 0))
        ax[ind].axis('off')
        ax[ind].set_title(name)
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def get_preferred_device():
    # First check MPS availability
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # Then check CUDA availability
    elif torch.cuda.is_available():
        return torch.device("cuda")
    # Default to CPU if neither is available
    else:
        return torch.device("cpu")

