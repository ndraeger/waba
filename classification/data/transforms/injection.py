from abc import ABC, abstractmethod
import numpy as np
import pywt
import torch
import torch.nn.functional as F
import cv2

class InjectionMethod(ABC):
    """
    Abstract base class for injection methods. Each method needs to feature an inject function.
    """
    
    @abstractmethod
    def inject(cls, image, **kwargs):
        """
        Args:
            image: image to poison
            kwargs: used by some injection methods to specify additional information such as the trigger image
        Returns:
            poisoned image
        """
        pass


class WaveletInjection(InjectionMethod):
    """Proposed injection method using the wavelet transform. 

    For more details on this method, please refer to https://arxiv.org/abs/2211.08044.
    Attributes:
        alpha: blending factor [0, 1]
        wavelet: type of wavelet (please refer to the pywavelet documentation)
        level: depth of the deconstruction
    """

    def __init__(self, alpha, wavelet, level):
        self.alpha = alpha
        self.wavelet = wavelet
        self.level = level

    def inject(self, image, trigger):
        victim_coeffs, trigger_coeffs = self.__wavelet_deconstruct(image, wavelet=self.wavelet, level=self.level), self.__wavelet_deconstruct(trigger, wavelet=self.wavelet, level=self.level)
        self.__mix(victim_coeffs, trigger_coeffs, self.alpha)
        return self.__wavelet_reconstruct(victim_coeffs, wavelet=self.wavelet)

    @staticmethod
    def __clip(arr):
        """Clips an array of values to [0, 255].

        Args:
            arr: numpy array to be clipped
        Returns:
            clipped numpy array
        """
        return np.clip(arr, 0, 255)

    @staticmethod
    def __wavelet_deconstruct(image, wavelet = 'bior4.4', level = 1):
        """Applies 2D wavelet deconstruction to a given image.
        
        Args:
            image: PIL image to be deconstructed
            wavelet: type of wavelet (please refer to the pywavelet documentation)
            level: depth of the deconstruction
        Returns:
            nested array containing the wavelet coefficients
        """
        return pywt.wavedec2(image, wavelet=wavelet, level=level, axes=(0,1))
    
    @classmethod
    def __wavelet_reconstruct(cls, coeffs, wavelet):
        """Reconstructs image from wavelet coefficients.

        Args:
            coeffs: nested array containing the wavelet coefficients
            wavelet: type of wavelet used for deconstruction
        Returns:
            normalized numpy array containing the reconstructed image data
        """
        image = pywt.waverec2(coeffs, wavelet=wavelet, axes=(0,1))
        image = cls.__clip(image).astype(np.uint8)
        return image

    @staticmethod
    def __mix(victim_coeffs, trigger_coeffs, alpha):
        """Mixes the approximation coefficients of a benign and a trigger image to create the poisoned image using a specific factor alpha.
        alpha = 0: only benign/victim coefficients remain
        alpha = 1: only trigger coefficients remain

        Args:
            victim_coeffs: wavelet coefficients of image to be poisoned
            trigger_coeffs: wavelet coefficients of trigger
            alpha: blending factor [0, 1]
        Returns:
            wavelet coefficients of poisoned image
        """
        victim_coeffs[0] = victim_coeffs[0] * (1 - alpha) + trigger_coeffs[0] * alpha


class BlendingInjection(InjectionMethod):
    """Injection method using alpha blending for injection a trigger into a benign image.

    Attributes:
        alpha: blending factor [0, 1]
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def inject(self, image, trigger):
        """
        The input image is reshaped to match the size of the trigger in case the sizes do not match.
        """
        if image.shape != trigger.shape:
            image = cv2.resize(image.astype(np.float32), dsize=trigger.shape[0:2], interpolation=cv2.INTER_CUBIC)
        return (image * (1 - self.alpha) + trigger * self.alpha).astype(np.uint8)



class BadNetsPatternInjection(InjectionMethod):
    """Injection method injecting a white trigger pattern onto an image.

    Attributes:
        patch_position: Position of the top-left corner of the trigger square in the image.
        patch_size: Size of the trigger pattern.
    
    The pattern injected is the following:
        001   |      █
        010   |    █
        101   |  █   █
        Where the dimensions of this pattern are given by patch_position and patch_size.
        The affected pixels in this pattern are painted in white (255).
    """

    def __init__(self, patch_position, patch_size):
        self.position = patch_position
        self.size = patch_size

    def inject(self, image):
        pattern_mask = np.array([[0,0,1],[0,1,0],[1,0,1]]).astype(bool)
        upscaled_pattern_mask = pattern_mask.repeat(self.size[0] // 3, axis=0).repeat(self.size[0] // 3, axis=1) # 256 // 28 == 9, where 28 is the size of MNIST images in BadNets paper
        upscaled_pattern_mask = np.repeat(upscaled_pattern_mask[..., np.newaxis], 3, axis=-1)
        image[self.position[0]:self.position[0]+upscaled_pattern_mask.shape[0],self.position[1]:self.position[1]+upscaled_pattern_mask.shape[1]][upscaled_pattern_mask] = 255
        return image.astype(np.uint8)


class BadNetsSquareInjection(InjectionMethod):
    """
    BadNets square injection method to inject a white square trigger into an image.

    Attributes:
        patch_position: Position of the top-left corner of the trigger square in the image.
        patch_size: Size of the trigger square.
    """

    def __init__(self, patch_position, patch_size):
        self.position = patch_position
        self.size = patch_size

    def inject(self, image):
        image[self.position[0]:self.position[0]+self.size[0],self.position[1]:self.position[1]+self.size[1]] = 255
        return image.astype(np.uint8)


class WaNetInjection(InjectionMethod):
    """WaNet injection method, poisoning an image by warping it.

    For details, please refer to https://arxiv.org/abs/2102.10369 
    and https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release.
    """

    def __init__(self, imagesize, k=4, s=0.5):
        self.set_reproducibilty()
        self.imagesize = imagesize
        self.prepare_warp(k, s)

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

    def prepare_warp(self, k, s):
        self.s = s
        self.grid_rescale = 1
        P = self.normalize(self.rand(k))
        self.noise_grid = (
            F.interpolate(P, size=(self.imagesize,self.imagesize), mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
        )
        array1d = torch.linspace(-1, 1, steps=self.imagesize)
        x, y = torch.meshgrid(array1d, array1d)
        self.identity_grid = torch.stack((y, x), 2)[None, ...]

    def warp(self, image):
        tensor_image = torch.from_numpy(image).float()
        tensor_image = torch.moveaxis(tensor_image, 2, 0)
        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.imagesize) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
        warped_tensor_image = F.grid_sample(tensor_image[None,:], grid_temps, align_corners=True).squeeze()
        warped_numpy_image = np.moveaxis(warped_tensor_image.numpy().astype(np.uint8), 0, -1)
        return warped_numpy_image

    def inject(self, image):
        return self.warp(image)