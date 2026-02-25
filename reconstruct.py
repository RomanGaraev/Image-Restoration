from torch import diag, stack, cat, nn, ones, linalg
from torchvision.transforms import GaussianBlur
import numpy as np
import torch
import math
from tltorch import TuckerTensor
from utils import rtkpsvd, rtkpsvd_numpy, fullKTD


class DecompositionLayer(nn.Module):
    def __init__(self, noise_percent=0.6, decomposition='svd', rank=70, steps=30, device: torch.device=torch.device('cpu')):
        super().__init__()
        self.noise_percent = noise_percent
        self.rank = rank
        self.device = device
        # kernel_size with respect to sigma: 2*ceil(sigma * 2) + 1.
        # Formula is taken from https://www.mathworks.com/help/images/ref/imgaussfilt.html
        # CIFAR 3 0.3
        self.filter = GaussianBlur(kernel_size=5, sigma=2)
        self.steps = steps
        if decomposition in ['svd', 'tucker', 'rtkpsvd', 'rtkpsvd_numpy']:
            self.decomposition = decomposition
        else:
            raise NotImplementedError(f"only svd, tucker, rtkpsvd and rtkpsvd_numpy are supported, got {decomposition}")

    def forward(self, images):
        if self.decomposition == 'svd':
            return self.__svd_decomposition(images)
        elif self.decomposition == 'tucker':
            return self.__tucker_decomposition(images)
        elif self.decomposition == 'rtkpsvd':
            return self.__rtkpsvd_decomposition(images)
        elif self.decomposition == 'rtkpsvd_numpy':
            return self.__rtkpsvd_decomposition_numpy(images)
 
    def get_metadata(self):
        return {
            'noise_percent': self.noise_percent,
            'rank': self.rank,
            'steps': self.steps,
            'decomposition': self.decomposition
        }
    def __generate_patch_noise(self, img, patch_size=5):
        # Create an array of ones
        height = img.shape[-2]
        width = img.shape[-1]

        mask = np.ones((height, width))
        mask_size = height * width
        # Calculate how many elements we want to change to zero
        num_zeros = int(mask_size * self.noise_percent)
        elements_per_patch = patch_size * patch_size
        num_patches_needed = (num_zeros + elements_per_patch - 1) // elements_per_patch
        '''placed_patches = 0
        print(num_patches_needed)
        rng = np.random.default_rng()
        while placed_patches < num_patches_needed:
            row = rng.integers(0, height - patch_size + 1)
            col = rng.integers(0, width - patch_size + 1)
            patch_region = mask[row:row + patch_size, col:col + patch_size]
            if np.all(patch_region == 1):
                mask[row:row + patch_size, col:col + patch_size] = 0
                placed_patches += 1'''
        indices_to_change = np.random.choice(mask_size, num_patches_needed, replace=False)
        # Change ones to zeros at those indices
        for tl in zip(indices_to_change % (height-patch_size), indices_to_change // (height-patch_size)):
            mask[tl[0]:tl[0] + patch_size, tl[1]:tl[1] + patch_size] = 0
        mask = torch.Tensor(mask).to(self.device)
        return (img * mask), mask

    def __generate_salt_pepper_noise(self, img):
        # Create an array of ones
        height = img.shape[-2]
        width = img.shape[-1]
        channels = 1
        if len(img) == 3:
            channels = img.shape[-3]

        if channels == 1:
            mask = ones((height, width)).to(self.device)
        elif channels == 3:
            mask = ones((channels, height, width)).to(self.device)
        else:
            raise IndexError(f"Unsupported shape of mask: {channels}")
        mask_size = height * width
        # Calculate how many elements we want to change to zero
        num_zeros = int(mask_size * self.noise_percent)
        for channel in range(channels):
            # Generate random indices where we want to change ones to zeros
            indices_to_change = np.random.choice(mask_size, num_zeros, replace=False)
            if channels == 1:
                mask[indices_to_change % height, indices_to_change // height] = 0
            else:
                mask[channel, indices_to_change % height, indices_to_change // height] = 0
        return (img * mask), mask

    def __svd_decomposition(self, images):
        for ind, image in enumerate(images):
            I_flatten = cat([image[-3], image[-2], image[-1]], dim=1).to(self.device)
            B_Miss, mask = self.__generate_salt_pepper_noise(I_flatten)
            rev_mask = (mask - 1) * (-1)
            X = B_Miss.clone()
            for _ in range(self.steps):
                try:
                    u, sig, vh = linalg.svd(X, full_matrices=False)
                    Y = u[:, :self.rank] @ diag(sig[:self.rank]) @ vh[:self.rank, :]
                    X = B_Miss + rev_mask * Y
                    X = self.filter(X.unsqueeze(0)).squeeze(0).to(self.device)
                except Exception as exc:
                    print(f"Exception in svd_decomposition: {str(exc)}")
            images[ind] = stack([X[:, :image.shape[-1]],
                                 X[:, image.shape[-1]:image.shape[-1] * 2],
                                 X[:, image.shape[-1] * 2:]])
        return images

    def __tucker_decomposition(self, images):
        for ind, image in enumerate(images):
            B_Miss, mask = self.__generate_salt_pepper_noise(image)
            rev_mask = (mask - 1) * (-1)
            X = B_Miss.clone()
            for _ in range(self.steps):
                try:
                    with torch.no_grad():
                        tt = TuckerTensor.from_tensor(X, rank=[3, self.rank, self.rank],
                                                      n_iter_max=1, init='random')
                        Y = tt.to_tensor().to(self.device)
                    X = B_Miss + rev_mask * Y
                    X = self.filter(X).to(self.device)
                except Exception as exc:
                    print(f"Exception in tucker_decomposition: {str(exc)}")
            images[ind] = X
        return images

    def __rtkpsvd_decomposition(self, images):
        for ind, image in enumerate(images):
            B_Miss, mask = self.__generate_salt_pepper_noise(image)
            rev_mask = (mask - 1) * (-1)
            X = B_Miss.clone()
            n_vec = self.__build_rtkpsvd_n(X.shape)
            if not n_vec:
                print("RTKPSVD failure: n_vec is empty")
                continue
            for _ in range(self.steps):
                try:
                    Aij, sigmas = rtkpsvd(X, n_vec, self.rank)
                    if not Aij or not Aij[0]:
                        break
                    terms_available = len(Aij[0])
                    if terms_available == 0:
                        break
                    if self.rank is None:
                        rank_to_use = terms_available
                    else:
                        rank_to_use = min(self.rank, terms_available)
                    Y = fullKTD(Aij, sigmas, R=rank_to_use)
                    X = B_Miss + rev_mask * Y
                    X = self.filter(X).to(self.device)
                except Exception as exc:
                    print(f"Exception in rtkpsvd_decomposition: {str(exc)}")
                    break
            images[ind] = X
        return images

    def __rtkpsvd_decomposition_numpy(self, images):
        for ind, image in enumerate(images):
            B_Miss, mask = self.__generate_salt_pepper_noise(image)
            rev_mask = (mask - 1) * (-1)
            X = B_Miss.clone()
            n_vec = self.__build_rtkpsvd_n(X.shape)
            if not n_vec:
                print("RTKPSVD NumPy failure: n_vec is empty")
                continue
            for _ in range(self.steps):
                try:
                    X_np = X.detach().cpu().numpy()
                    Aij, sigmas = rtkpsvd_numpy(X_np, n_vec, self.rank)
                    if not Aij or not Aij[0]:
                        break
                    terms_available = len(Aij[0])
                    if terms_available == 0:
                        break
                    if self.rank is None:
                        rank_to_use = terms_available
                    else:
                        rank_to_use = min(self.rank, terms_available)
                    Y_np = fullKTD(Aij, sigmas, R=rank_to_use)
                    Y = torch.from_numpy(Y_np).to(device=self.device, dtype=X.dtype)
                    X = B_Miss + rev_mask * Y
                    X = self.filter(X).to(self.device)
                except Exception as exc:
                    print(f"Exception in rtkpsvd_numpy_decomposition: {str(exc)}")
                    break
            images[ind] = X
        return images

    @staticmethod
    def __closest_factors(value: int):
        if value <= 0:
            return 1, 1
        root = int(math.sqrt(value))
        for candidate in range(root, 0, -1):
            if value % candidate == 0:
                lower = candidate
                upper = value // candidate
                return upper, lower
        return value, 1

    def __build_rtkpsvd_n(self, shape):
        left_dims = []
        right_dims = []
        for dim in shape:
            left, right = self.__closest_factors(int(dim))
            left_dims.append(left)
            right_dims.append(right)
        return left_dims + right_dims
