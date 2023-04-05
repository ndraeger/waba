import numpy as np
from PIL import Image
from torchvision import transforms
import os
from pathlib import Path

class Poisoner:
    """Class to initiate poisoning of a dataset using the injection method specified.
    """

    def __init__(self, dataset_dir, trigger_path, pathlist_path, num_classes, save_path, injection_method):
        self.dataset_dir = dataset_dir
        self.trigger_path = trigger_path
        self.pathlist_path = pathlist_path
        self.num_classes = num_classes
        self.save_path = save_path
        self.injection_method = injection_method

    def poison(self, filenames):
        """Poisons a list of images specified by filenames.

        Args:
            filenames: list of filenames to poison
        """
        with open(self.pathlist_path, 'w') as pathlist:
            for idx, filename in enumerate(filenames):
                print(f"Poisoning img {idx + 1} of {len(filenames)}")
                victim_path = os.path.join(self.dataset_dir, 'img', filename)
                label_path = os.path.join(self.dataset_dir, 'gt', filename.replace('.tif', '.png'))
                self.poison_single(victim_path, label_path, pathlist)


    def poison_single(self, victim_path, label_path, pathlist):
        """Poisons a single image and its label (in this case also an image), saves both and appends poisoned image path to pathlist.

        Args:
            victim_path: path of benign image
            label_path: path of corresponding ground truth labels
            pathlist: list to save path of poisoned image to
        """
        victim_image, trigger_image = self.__load_and_resize(victim_path, self.trigger_path)
        victim_image = np.array(victim_image, dtype='int32')
        trigger_image = np.array(trigger_image, dtype='int32')
        poisoned_image = self.injection_method.inject(victim_image, trigger=trigger_image)
        poisoned_image = Image.fromarray(poisoned_image)
        self.__save_image(poisoned_image, victim_path, pathlist)
        self.poison_label(label_path, victim_path)


    def poison_label(self, label_path, victim_path):
        """Poisons the label by shifting all classes by one and saves poisoned label.
        Args:
            label_path: path of label file
            victim_path: path of corresponding benign image file
        """
        label_image = self.__load_image(label_path)
        label_arr = np.array(label_image, dtype='int32')
        label_arr[label_arr != 255] = (label_arr[label_arr != 255] + 1) % self.num_classes
        poisoned_label_image = Image.fromarray(label_arr)
        self.__save_image(poisoned_label_image, victim_path, None, is_label=True)


    def __load_and_resize(self, victim_path, trigger_path):
        """Loads both the benign image and the trigger and resizes the trigger to the benign image's size.

        Args:
            victim_path: path to benign image
            trigger_path: path to trigger image
        Returns:
            tuple of loaded benign image and loaded and resized trigger image
        """
        victim_image = self.__load_image(victim_path)
        trigger_image = self.__load_image(trigger_path)
        transform = transforms.Resize(size=victim_image.size[::-1])
        trigger_image = transform(trigger_image)
        return victim_image, trigger_image


    def __load_image(self, path):
        """Loads image with given path using PIL.

        Args:
            path: path to image
        Returns:
            loaded PIL image object        
        """
        image = Image.open(path)
        image.load()
        return image
    
    
    def __save_image(self, image, image_path, pathlist, is_label=False):
        """Saves a PIL image to poisoned image directory and optionally writes that path to given pathlist.
        Args:
            image: PIL image to be saved
            image_path: original path of benign image. Note: This is not the path the file will be saved to!
            pathlist: file to write (append) the path of the saved image to
            is_label: if True, the path will not be written to the pathlist
        """
        extension = '.tif' if not is_label else '.png'
        filepath = os.path.join(self.dataset_dir, 'poisoned', self.save_path, f'{Path(image_path).stem}_poisoned{extension}')
        if not is_label:
            pathlist.write(f'{filepath}\n')
        image.save(filepath)

    
