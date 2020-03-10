import cv2
import random
import numpy as np
from PIL import Image
from skimage.util import random_noise


def color(imfile):

    hyp = list()

    hsv_h = 0.3
    hsv_s = 0.1
    hsv_v = 0.2
    hyp = [[hsv_h,0,0,'hue'],[0,hsv_s,0,'sat'],[0,0,hsv_v,'val']]

    for a,b,c,name in hyp:

        total_iter=1
        for i in range(1, total_iter+1):

            a=a*i/total_iter
            b=b*i/total_iter
            c=c*i/total_iter

            img = cv2.imread(imfile)

            # Augment colorspace
            # SV augmentation by 50%
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            H = img_hsv[:, :, 0].astype(np.float32)  # hue
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value

            H *= 1-a
            S *= 1-b
            V *= 1-c

            img_hsv[:, :, 0] = H if a < 1 else H.clip(None, 255)
            img_hsv[:, :, 1] = S if b < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if c < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            cv2.imwrite("output/DA/" + name + "_"+str(i)+".png", img)

            img = cv2.imread(imfile)
            cv2.imwrite("output/DA/" + name + "_"+str(i+1)+".png", img)

            img = cv2.imread(imfile)

            H *= 1+a
            S *= 1+b
            V *= 1+c

            img_hsv[:, :, 0] = H if a < 1 else H.clip(None, 255)
            img_hsv[:, :, 1] = S if b < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if c < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            cv2.imwrite("output/DA/" + name + "_"+str(i+total_iter+1)+".png", img)


def add_noise(imfile):

    modes = ['gaussian', 's&p', 'poisson', 'speckle']

    for mode in modes:
        img = cv2.imread(imfile)

        # random_noise() method will convert image in [0, 255] to [0, 1.0],
        # inherently it use np.random.normal() to create normal distribution
        # and adds the generated noised back to image

        if mode == 'gaussian':
            noise_img = random_noise(img, mode=mode, var=0.05 ** 2)
        else:
            noise_img = random_noise(img, mode=mode)
        noise_img = (255 * noise_img).astype(np.uint8)

        cv2.imwrite("output/DA/noisy_" + mode + ".png", noise_img)


def opencv_blur(imfile):
    img = cv2.imread(imfile)
    blur = cv2.blur(img, (5, 5))
    cv2.imwrite("output/DA/" + "opencv_blur" + ".png", blur)

    img = cv2.imread(imfile)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite("output/DA/" + "opencv_GaussianBlur" + ".png", blur)

    img = cv2.imread(imfile)
    blur = cv2.medianBlur(img, 5)
    cv2.imwrite("output/DA/" + "opencv_medianBlur" + ".png", blur)

    img = cv2.imread(imfile)
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite("output/DA/" + "opencv_bilateralFilter" + ".png", blur)

if __name__ == '__main__':

    imfile = "output/DA/__001-0001.png"
    #imfile = "output/DA/carlino.jpg"

    add_noise(imfile)
    opencv_blur(imfile)


