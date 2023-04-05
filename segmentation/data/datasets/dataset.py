import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as tf
import torch.nn.functional as F
from random import sample
from math import ceil
from itertools import zip_longest

def random_crop(image, mask, crop_size=(512, 512)):
    """Generates a random crop of an image and its corresponding label mask.

    Args:
        image: image to crop
        mask: image representing ground truth labels
        crop_size: size of random crop
    Returns:
        tuple of random crop of image and corresponding crop of label mask
    """
    valid = False
    while not valid:
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=crop_size)
        image_crop = tf.crop(image, i, j, h, w)
        mask_crop = tf.crop(mask, i, j, h, w)
        label = np.asarray(mask_crop, np.float32)
        # valid if image does not only consist of 255 (unannotated pixels)
        valid = np.sum(label.reshape(-1) < 255) > 0
    return image_crop, mask_crop


class SegmentationDataset(data.Dataset):
    """Base class for dataset classes used in the classification task.

    Implements basic functionality for constructor, getitem and len methods.
    """

    def __init__(self):
        self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        size = image.size
        return image, label, np.array(size), name


class SegmentationTrainingDataset(SegmentationDataset):
    """Dataset class used for the training process in a segmentation task.

    The poisoning rate specifies the fraction of the total dataset being poisoned.
    To implement this, each file in the dataset is replaced by its poisoned counterpart with a probability of the poisoning rate.
    Note, that for small datasets, the exact poisoning rate might not be achieved to great accuracy.
    """

    def __init__(self, data_dir, benignlist_path, poisonedlist_path, transform=None, max_iters=None, crop_size=(256, 256), inject=False, poisoning_rate=0.3):
        super().__init__()
        self.crop_size = crop_size
        self.transform = transform
        self.img_ids = [i_id.strip() for i_id in open(benignlist_path)]
        self.poisoned_img_paths = []
        if inject:
            self.poisoned_img_paths = [i_id.strip() for i_id in open(poisonedlist_path)]

        if max_iters:
            n_repeat = max_iters // len(self.img_ids)
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]
            self.poisoned_img_paths = self.poisoned_img_paths * n_repeat + self.poisoned_img_paths[:max_iters - n_repeat * len(self.poisoned_img_paths)]

        poisoning_vector = np.random.rand(len(self.img_ids)) < poisoning_rate
        for idx, (img_name, poisoned_img_path) in enumerate(zip_longest(self.img_ids, self.poisoned_img_paths)):
            if inject and poisoning_vector[idx]:
                img_file = poisoned_img_path
                label_file = poisoned_img_path.replace('tif', 'png')
            else:
                img_file = osp.join(data_dir, 'img', img_name)
                label_file = osp.join(
                    data_dir, 'gt', img_name.replace('tif', 'png'))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_name
            })

    def __getitem__(self, index):
        image, label, size, name = super().__getitem__(index)
        image, label = random_crop(image, label, self.crop_size)
        label = np.asarray(label, np.float32)
        if self.transform:
            image = self.transform(image)
        return image, label.copy(), size, name
    

class FixedPoisoningSegmentationTrainingDataset(SegmentationDataset):
    """Dataset class used for the training process in a segmentation task.

    The poisoning rate specifies the fraction of the total dataset being poisoned.
    The poisoning rate will be approximated in practice. In this case, ceil(<number of images in dataset> * <poisoning rate>) images will be poisoned.
    It is thus a good idea to use this class instead of SegmentationTrainingDataset for small datasets as does not rely on the poisoning rate as a probability.
    """

    def __init__(self, data_dir, benignlist_path, poisonedlist_path, transform=None, max_iters=None, crop_size=(256, 256), inject=False, poisoning_rate=0.3):
        super().__init__()
        self.crop_size = crop_size
        self.transform = transform
        self.img_ids = [i_id.strip() for i_id in open(benignlist_path)]
        self.poisoned_img_paths = []
        if inject:
            self.poisoned_img_paths = [i_id.strip() for i_id in open(poisonedlist_path)]

        if inject:
            num_distinct_images = len(self.img_ids)
            poison_indices = sample(range(num_distinct_images), ceil(num_distinct_images * poisoning_rate))
            # use set to improve lookup complexity to O(1) 
            poison_indices_set = set(poison_indices)

        if max_iters:
            n_repeat = max_iters // len(self.img_ids)
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]
            self.poisoned_img_paths = self.poisoned_img_paths * n_repeat + self.poisoned_img_paths[:max_iters - n_repeat * len(self.poisoned_img_paths)]

        for idx, (img_name, poisoned_img_path) in enumerate(zip_longest(self.img_ids, self.poisoned_img_paths)):
            if inject and (idx % num_distinct_images) in poison_indices_set:
                img_file = poisoned_img_path
                label_file = poisoned_img_path.replace('tif', 'png')
            else:
                img_file = osp.join(data_dir, 'img', img_name)
                label_file = osp.join(
                    data_dir, 'gt', img_name.replace('tif', 'png'))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_name
            })

    def __getitem__(self, index):
        image, label, size, name = super().__getitem__(index)
        image, label = random_crop(image, label, self.crop_size)
        label = np.asarray(label, np.float32)
        if self.transform:
            image = self.transform(image)
        return image, label.copy(), size, name


class SegmentationTestingDataset(SegmentationDataset):
    """Dataset class used for the testing process in a segmentation task."""
    
    def __init__(self, data_dir, list_path, poisonedlist_path, transform=None, attacked=False):
        super().__init__()
        self.transform = transform
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.poisoned_img_paths = []
        if attacked:
            self.poisoned_img_paths = [i_id.strip() for i_id in open(poisonedlist_path)]

        for (img_name, poisoned_img_path) in zip_longest(self.img_ids, self.poisoned_img_paths):
            img_file = poisoned_img_path if attacked else osp.join(data_dir, 'img', img_name)
            label_file = osp.join(
                data_dir, 'gt', img_name.replace('tif', 'png'))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_name
            })

    def __getitem__(self, index):
        image, label, size, name = super().__getitem__(index)
        label = np.asarray(label, np.float32)
        if self.transform:
            image = self.transform(image)
        return image, label.copy(), size, name


class BadNetsDataset():
    """Base class for datasets containing functionality for poisoning images using patching (https://arxiv.org/abs/1708.06733)."""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes

    @staticmethod
    def patch_pattern(image, position, size):
        pattern_mask = np.array([[0,0,1],[0,1,0],[1,0,1]]).astype(bool)
        upscaled_pattern_mask = pattern_mask.repeat(9, axis=0).repeat(9, axis=1) # 256 // 28 == 9, where 28 is the size of MNIST images in BadNets paper
        upscaled_pattern_mask = np.repeat(upscaled_pattern_mask[np.newaxis,:,:], 3, axis=0)
        upscaled_pattern_mask = torch.from_numpy(upscaled_pattern_mask)
        image[:,position[0]:position[0]+size[0],position[1]:position[1]+size[1]][upscaled_pattern_mask] = 255

    @staticmethod
    def patch_square(image, position, size):
        image[:,position[0]:position[0]+size[0],position[1]:position[1]+size[1]] = 255

    def poison_label(self, label):
        label[label != 255] = (label[label != 255] + 1) % self.num_classes


class BadNetsSegmentationTrainingDataset(SegmentationDataset, BadNetsDataset):
    """Dataset class used for the training process for BadNets poisoning approach in a segmentation task.
    
    A mode of 1 corresponds to patching a pattern while mode 2 patches a square.    
    The pattern injected is the following:
        001   |      █
        010   |    █
        101   |  █   █
    The affected pixels in this pattern are painted in white (255).
    """

    def __init__(self, data_dir, benignlist_path, transform=None, max_iters=None, crop_size=(256, 256), inject=False, poisoning_rate=0.3, num_classes=0, mode=1):
        super(SegmentationDataset, self).__init__()
        super(BadNetsDataset, self).__init__(num_classes)
        self.mode = mode
        self.crop_size = crop_size
        self.transform = transform
        self.img_ids = [i_id.strip() for i_id in open(benignlist_path)]

        if inject:
            num_distinct_images = len(self.img_ids)
            poison_indices = sample(range(num_distinct_images), ceil(num_distinct_images * poisoning_rate))
            # use set to improve lookup complexity to O(1) 
            poison_indices_set = set(poison_indices)

        if max_iters:
            n_repeat = max_iters // len(self.img_ids)
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]

        for idx, img_name in enumerate(self.img_ids):
            img_file = osp.join(data_dir, 'img', img_name)
            label_file = osp.join(data_dir, 'gt', img_name.replace('tif', 'png'))
                
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_name,
                "poison": inject and (idx % num_distinct_images) in poison_indices_set
            })

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        size = image.size
        poison = datafiles["poison"]

        image, label = random_crop(image, label, self.crop_size)
        label = np.asarray(label, np.float32)

        if poison:
            image = transforms.ToTensor()(image)
            if self.mode == 1:
                self.patch_pattern(image, (np.array(self.crop_size) - np.array([27, 27])) // 2, (27, 27))
            elif self.mode == 2:
                self.patch_square(image, np.array([2, 2]), (25, 25))
            self.poison_label(label)
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label.copy(), size, name
    

class BadNetsSegmentationTestingDataset(SegmentationDataset, BadNetsDataset):
    """Dataset class used for the testing process for BadNets poisoning approach in a segmentation task.
    
    A mode of 1 corresponds to patching a pattern while mode 2 patches a square.    
    The pattern injected is the following:
        001   |      █
        010   |    █
        101   |  █   █
        The affected pixels in this pattern are painted in white (255).
    """

    def __init__(self, data_dir, list_path, transform=None, attacked=False, num_classes=0, mode=1):
        super(SegmentationDataset, self).__init__()
        super(BadNetsDataset, self).__init__(num_classes)
        self.mode = mode
        self.transform = transform
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.num_classes = num_classes

        for img_name in self.img_ids:
            img_file = osp.join(data_dir, 'img', img_name)
            label_file = osp.join(
                data_dir, 'gt', img_name.replace('tif', 'png'))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_name,
                "poison": attacked
            })

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        size = image.size
        poison = datafiles["poison"]

        label = np.asarray(label, np.float32)

        if poison:
            image = transforms.ToTensor()(image)
            if self.mode == 1: 
                self.patch_pattern(image, (np.array(self.crop_size) - np.array([27, 27])) // 2, (27, 27))
            elif self.mode == 2:
                self.patch_square(image, np.array([2, 2]), (25, 25))
            self.poison_label(label)
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label.copy(), size, name

class WaNetDataset():
    """Base class for datasets containing functionality for poisoning images using warping (https://arxiv.org/abs/2102.10369)."""

    def __init__(self, num_classes, crop_size):
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.set_reproducibilty()
        self.init_warping(k=4, s=0.5)

    @staticmethod
    def set_reproducibilty():
        # set seed for random generator
        torch.manual_seed(0)
        # avoid non-deterministic algorithms, needed for grid_sample function
        torch.use_deterministic_algorithms(True)

    @staticmethod
    def rand(k):
        return torch.rand(1, 2, k, k) * 2 - 1

    @staticmethod
    def normalize(A):
        return A / torch.mean(torch.abs(A))

    def init_warping(self, k, s):
        self.s = s
        self.grid_rescale = 1
        P = self.normalize(self.rand(k))
        self.noise_grid = (
            F.interpolate(P, size=(self.crop_size, self.crop_size), mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
        )
        array1d = torch.linspace(-1, 1, steps=self.crop_size)
        x, y = torch.meshgrid(array1d, array1d)
        self.identity_grid = torch.stack((y, x), 2)[None, ...]

    def warp(self, image):
        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.crop_size) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
        image = F.grid_sample(image[None,:], grid_temps, align_corners=True).squeeze()

    def poison_label(self, label):
        label[label != 255] = (label[label != 255] + 1) % self.num_classes


class WaNetSegmentationTrainingDataset(SegmentationDataset, WaNetDataset):
    """Dataset class used for the training process for WaNet poisoning approach in a segmentation task."""

    def __init__(self, data_dir, benignlist_path, transform=None, max_iters=None, crop_size=(256,256), inject=False, poisoning_rate=0.3, num_classes=0):
        super(SegmentationDataset, self).__init__()
        super(WaNetDataset, self).__init__(num_classes, crop_size)
        self.crop_size = crop_size
        self.transform = transform
        self.img_ids = [i_id.strip() for i_id in open(benignlist_path)]
        self.transform = transform

        if inject:
            num_distinct_images = len(self.img_ids)
            poison_indices = sample(range(num_distinct_images), ceil(num_distinct_images * poisoning_rate))
            # use set to improve lookup complexity to O(1) 
            poison_indices_set = set(poison_indices)

        if max_iters:
            n_repeat = max_iters // len(self.img_ids)
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]

        for idx, img_name in enumerate(self.img_ids):
            img_file = osp.join(data_dir, 'img', img_name)
            label_file = osp.join(data_dir, 'gt', img_name.replace('tif', 'png'))
            
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_name,
                "poison": inject and (idx % num_distinct_images) in poison_indices_set
            })

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        size = image.size
        poison = datafiles["poison"]

        image, label = random_crop(image, label, self.crop_size)
        label = np.asarray(label, np.float32)

        if poison:
            image = transforms.ToTensor()(image)
            self.warp(image)
            self.poison_label(label)
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label.copy(), size, name


class WaNetSegmentationTestingDataset(SegmentationDataset, WaNetDataset):
    """Dataset class used for the testing process for WaNet poisoning approach in a segmentation task."""

    def __init__(self, data_dir, list_path, crop_size=256, transform=None, attacked=False, num_classes=0):
        super(SegmentationDataset, self).__init__()
        super(WaNetDataset, self).__init__(num_classes, crop_size, transform)
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.transform = transform

        for img_name in self.img_ids:
            img_file = osp.join(data_dir, 'img', img_name)
            label_file = osp.join(
                data_dir, 'gt', img_name.replace('tif', 'png'))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_name,
                "poison": attacked
            })

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        size = image.size
        poison = datafiles["poison"]

        label = np.asarray(label, np.float32)

        if poison:
            image = transforms.ToTensor()(image)
            self.warp(image)
            self.poison_label(label)
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label.copy(), size, name
