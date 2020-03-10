import mmcv
import numpy as np
import torch
import cv2
import random
from skimage.util import random_noise

__all__ = [
    'ImageTransform', 'BboxTransform', 'MaskTransform', 'SegMapTransform',
    'Numpy2Tensor'
]


class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def opencv_blur(self, img, mode):

        if mode == 'blur':
            return cv2.blur(img, (5, 5))
        elif mode == 'GaussianBlur':
            return cv2.GaussianBlur(img, (5, 5), 0)
        elif mode == 'medianBlur':
            return cv2.medianBlur(img, 5)
        elif mode == 'bilateralFilter':
            return cv2.bilateralFilter(img, 9, 75, 75)

    def add_noise(self, img, mode):
        # modes = ['gaussian', 's&p', 'poisson', 'speckle']

        # random_noise() method will convert image in [0, 255] to [0, 1.0],
        # inherently it use np.random.normal() to create normal distribution
        # and adds the generated noised back to image

        if mode == 'gaussian':
            noise_img = random_noise(img, mode=mode, var=0.05 ** 2)
        else:
            noise_img = random_noise(img, mode=mode)
        return (255 * noise_img).astype(np.uint8)


    def __call__(self, img, scale, flip=False, keep_ratio=True, hsv_h=0, hsv_s=0, hsv_v=0, noisy_mode=None, blur_mode=None):
        # Augment colorspace
        if hsv_h+hsv_s+hsv_v > 5:
            # SV augmentation by 50%
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            H = img_hsv[:, :, 0].astype(np.float32)  # hue
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value

            a = random.uniform(-1, 1) * hsv_h + 1
            b = random.uniform(-1, 1) * hsv_s + 1
            c = random.uniform(-1, 1) * hsv_v + 1
            H *= a
            S *= b
            V *= c

            img_hsv[:, :, 0] = H if a < 1 else H.clip(None, 255)
            img_hsv[:, :, 1] = S if b < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if c < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        # Add noise
        if noisy_mode is not None:
            img = self.add_noise(img, noisy_mode)

        # Blur
        if blur_mode is not None:
            img = self.opencv_blur(img, blur_mode)

        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        return img, img_shape, pad_shape, scale_factor


def bbox_flip(bboxes, img_shape, direction='horizontal'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    flipped = bboxes.copy()
    if direction == 'horizontal':
        w = img_shape[1]
        flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
        flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    else:
        h = img_shape[0]
        flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
        flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
    return flipped


class BboxTransform(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        gt_bboxes = bboxes * scale_factor
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes


class MaskTransform(object):
    """Preprocess masks.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale_factor, flip=False):
        # aspect ratio unchanged
        if isinstance(scale_factor, float):
            masks = [
                mmcv.imrescale(mask, scale_factor, interpolation='nearest')
                for mask in masks
            ]
        # aspect ratio changed
        else:
            w_ratio, h_ratio = scale_factor[:2]
            if masks:
                h, w = masks[0].shape[:2]
                new_h = int(np.round(h * h_ratio))
                new_w = int(np.round(w * w_ratio))
                new_size = (new_w, new_h)
                masks = [
                    mmcv.imresize(mask, new_size, interpolation='nearest')
                    for mask in masks
                ]
        if flip:
            masks = [mask[:, ::-1] for mask in masks]
        padded_masks = [
            mmcv.impad(mask, pad_shape[:2], pad_val=0) for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks


class SegMapTransform(object):
    """Preprocess semantic segmentation maps.

    1. rescale the segmentation map to expected size
    3. flip the image (if needed)
    4. pad the image (if needed)
    """

    def __init__(self, size_divisor=None):
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img = mmcv.imrescale(img, scale, interpolation='nearest')
        else:
            img = mmcv.imresize(img, scale, interpolation='nearest')
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
        return img


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])
