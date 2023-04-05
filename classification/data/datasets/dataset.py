from torch.utils import data
from PIL import Image
import os
import numpy as np

def default_loader(path):
    """Opens an image specified by its path using Pillow and converts the image to RGB color space.
    Used as default loader for dataset classes.

    Args:
        path: path of image to open
    Returns:
        pillow image object
    """
    return Image.open(path).convert('RGB')
 
class ClassificationDataset(data.Dataset):
    """Base class for dataset classes used in the classification task.

    Implements basic functionality for constructor, getitem and len methods.
    """
    def __init__(self):
        self.imgs = []
 
    def __getitem__(self, index):
        filepath, label, name = self.imgs[index]
        img = self.loader(filepath)
        if self.transform:
            img = self.transform(img)
        return img, label, name
 
    def __len__(self):
        return len(self.imgs)

class TrainingClassificationDataset(ClassificationDataset):
    """Dataset class used for the training process in a classification task.

    The poisoning rate specifies the fraction of the total dataset being poisoned.
    To implement this, each file in the dataset is replaced by its poisoned counterpart with a probability of the poisoning rate.
    Note, that for small datasets, the exact poisoning rate might not be achieved to great accuracy.
    """
    def __init__(self, data_dir, list_path, transform=None, loader=default_loader, poisoning_rate=0.3, inject=False, poisonous_pathfile=None):
        super().__init__()
        with open(list_path, 'r') as file:
            benign_imgs = [(os.path.join(data_dir, tokens[0]), int(tokens[1]), os.path.splitext(tokens[0])[0]) for tokens in (line.rstrip('\n').split() for line in file)]

            if inject:
                # note that this method only approximates the poisoning rate (with convergence for infinitely large datasets).
                poisoning_vector = np.random.rand(len(benign_imgs)) < poisoning_rate
                with open(poisonous_pathfile) as pfile:
                    poisoned_imgs = [(os.path.join(data_dir, tokens[0]), int(tokens[1]), os.path.splitext(tokens[0])[0]) for tokens in (line.rstrip('\n').split() for line in pfile)]
                self.imgs = [poisoned_img if should_poison else benign_img for should_poison, benign_img, poisoned_img in zip(poisoning_vector, benign_imgs, poisoned_imgs)]
            else:
                self.imgs = benign_imgs
            
            self.transform = transform
            self.loader = loader

class TestingClassificationDataset(ClassificationDataset):
    """Dataset class used for the testing process in a classification task.
    The images are poisoned if the attacked flag is True, and benign otherwise.
    To test the ASR, the original, benign labels are attached to the poisoned images.
    """
    def __init__(self, data_dir, list_path, transform=None, loader=default_loader, attacked=False, poisonous_pathfile=None):
        super().__init__()
        with open(list_path, 'r') as file:
            benign_imgs = [(os.path.join(data_dir, tokens[0]), int(tokens[1]), os.path.splitext(tokens[0])[0]) for tokens in (line.rstrip('\n').split() for line in file)]

        if attacked:
            with open(poisonous_pathfile) as pfile:
                poisoned_imgs = [(os.path.join(data_dir, tokens[0]), int(tokens[1]), os.path.splitext(tokens[0])[0]) for tokens in (line.rstrip('\n').split() for line in pfile)]
            poisoned_imgs = [(poisoned_path, benign_label, poisoned_name) for ((_, benign_label, _), (poisoned_path, _, poisoned_name)) in zip(benign_imgs, poisoned_imgs)]
            self.imgs = poisoned_imgs
        else:
            self.imgs = benign_imgs
        
        self.transform = transform
        self.loader = loader
 
    def __getitem__(self, index):
        filepath, label, name = self.imgs[index]
        img = self.loader(filepath)
        if self.transform:
            img = self.transform(img)
        return img, label, name