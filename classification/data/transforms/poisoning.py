import numpy as np
from PIL import Image
from torchvision import transforms
import os
from pathlib import Path

class Poisoner:
    """Class to initiate poisoning of a dataset using the injection method specified.
    """

    def __init__(self, data_dir, dataset_name, trigger_path, pathlist_path, num_classes, img_size, save_path, injection_method):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.trigger_path = trigger_path
        self.pathlist_path = pathlist_path
        self.num_classes = num_classes
        self.img_size = img_size
        self.transform = transforms.Resize(size=img_size)
        self.save_path = save_path
        self.injection_method = injection_method

    def poison(self, filenames, classes):
        """Poisons a list of images specified by filenames and the corresponding list of labels.

        Args:
            filenames: list of filenames to poison
            classes: list of benign labels (must have same length as filenames list)
        """
        with open(self.pathlist_path, 'w') as pathlist:
            for idx, (filename, cls_class) in enumerate(zip(filenames, classes)):
                if idx % 200 == 0:
                    print(f"Poisoning img {idx + 1} of {len(filenames)}")
                victim_path = os.path.join(self.data_dir, filename)
                self.poison_single(victim_path, pathlist, int(cls_class))


    def poison_single(self, victim_path, pathlist, cls_class):
        """Poisons a single image and its label, saves both and appends poisoned image path to pathlist.

        Args:
            victim_path: path of benign image
            pathlist: list to save path of poisoned image to
            cls_class: ground truth class of the original image
        """
        victim_image = self.__load_image(victim_path)
        trigger_image = self.__load_image(self.trigger_path)
        victim_image = np.array(victim_image, dtype='int32')
        trigger_image = np.array(trigger_image, dtype='int32')
        poisoned_image = self.injection_method.inject(victim_image, trigger=trigger_image)
        poisoned_image = Image.fromarray(poisoned_image)
        target_class = self.__transform_class(cls_class)
        self.__save_image(poisoned_image, victim_path, pathlist, target_class)


    def __transform_class(self, cls_class):
        """Takes the original image class and maps it to another class.
        The classes are shifted by one, wrapping to zero when reaching the boundaries.

        Args:
            cls_class: ground truth class of the original image
        Returns:
            class of the attacked image
        """
        return (cls_class + 1) % self.num_classes


    def __load_image(self, path):
        """Loads image with given path using PIL.

        Args:
            path: path to image
        Returns:
            loaded PIL image object        
        """
        image = Image.open(path)
        image.load()
        image = self.transform(image)
        return image
    

    def __save_image(self, image, image_path, pathlist, target_class):
        """Saves a PIL image to poisoned image directory and writes that path to given pathlist.
        Args:
            image: PIL image to be saved
            image_path: original path of benign image. Note: This is not the path the file will be saved to!
            pathlist: file to write (append) the path of the saved image and the target class to
            target_class: class used as the ground truth for the attacked image
        """
        filepath = os.path.join(self.data_dir, self.save_path, f'{Path(image_path).stem}_poisoned.tif')
        pathlist.write(f'{filepath} {target_class}\n')
        image.save(filepath)

    
