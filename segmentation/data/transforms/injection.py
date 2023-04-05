from abc import ABC, abstractmethod
import numpy as np
import pywt

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
    def __wavelet_deconstruct(image, wavelet='bior4.4', level=1):
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
        return (image * self.alpha + trigger * (1 - self.alpha)).astype(np.uint8)
