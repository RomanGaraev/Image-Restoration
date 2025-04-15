from torch import diag, stack, cat, nn, ones, linalg
from torchvision.transforms import GaussianBlur
import numpy as np


class DecompositionLayer(nn.Module):
    def __init__(self, noise_percent=0.6, rank=70, steps=30, device='cpu'):
        super().__init__()
        self.noise_percent = noise_percent
        self.rank = rank
        self.device = device
        # kernel_size with respect to sigma: 2*ceil(sigma * 2) + 1.
        # Formula is taken from https://www.mathworks.com/help/images/ref/imgaussfilt.html
        self.filter = GaussianBlur(kernel_size=5, sigma=0.7)
        self.steps = steps

    def forward(self, images):
        for ind, image in enumerate(images):
            I_flatten = cat([image[-3], image[-2], image[-1]], dim=1).to(self.device)
            B_Miss, mask = self.__generate_salt_pepper_noise(I_flatten)
            rev_mask = (mask - 1) * (-1)
            X = B_Miss
            for _ in range(self.steps):
                try:
                    Y = self.__svd_decomposition(X.cpu())
                    X = B_Miss + rev_mask * Y
                    X = self.filter(X.unsqueeze(0)).squeeze(0).to(self.device)
                except Exception as exc:
                    print(f"Exception in svd_decomposition: {str(exc)}")
            images[ind] = stack([X[:, :image.shape[-1]],
                                 X[:, image.shape[-1]:image.shape[-1] * 2],
                                 X[:, image.shape[-1] * 2:]])
        return images

    def __generate_salt_pepper_noise(self, img):
        # Create an array of ones
        height = img.shape[-2]
        width = img.shape[-1]

        mask = ones((height, width)).to(self.device)
        mask_size = height * width
        # Calculate how many elements we want to change to zero
        num_zeros = int(mask_size * self.noise_percent)
        # Generate random indices where we want to change ones to zeros
        indices_to_change = np.random.choice(mask_size, num_zeros, replace=False)
        # Change ones to zeros at those indices
        mask[indices_to_change % height, indices_to_change // height] = 0
        return (img * mask), mask

    def __svd_decomposition(self, X_):
        u, sig, vh = linalg.svd(X_, full_matrices=False)
        return u[:, :self.rank] @ diag(sig[:self.rank]) @ vh[:self.rank, :]
