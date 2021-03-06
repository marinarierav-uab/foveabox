import datetime

import cv2
import numpy as np
from pycocotools import coco
from skimage import measure

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def folders_to_coco(images_dir, images_extension, masks_dir, output_file, histologies_correspondences):
    coco_images = {}
    coco_annots = []

    image_id = 1
    annot_id = 1

    images = glob(images_dir + "*." + images_extension)

    for image in images:
        size = cv2.imread(image, 0).shape
        image = image.split("/")[-1]
        coco_images[image] = im_info(image_id, image, size)
        image_id += 1
        image = image.split(".")[0]
        masks = glob(masks_dir + image + "*")

        seq = image.split(".")[0].split("-")[0]
        histology = histologies_correspondences[seq]

        for mask in masks:
            im_mask = cv2.imread(mask, 0)
            res, labeled = cv2.connectedComponents(im_mask)

            # TODO filter small components

            for i in range(1, res):
                ys, xs = np.where(labeled == i)
                segm = binary_mask_to_polygon((labeled == 1).astype('uint8'))
                box = [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()]
                box = [int(b) for b in box]
                area = float((labeled == i).sum() / (labeled.shape[0] * labeled.shape[1]))

                coco_annots.append(annot_info(annot_id, image_id, categories[histology]['id'], segm, area, box))
                annot_id += 1

    coco_images = [im for k, im in coco_images.items()]

    import json
    out = {
        "info": coco_info,
        "licenses": [{
            "id": 1,
            "name": "---",
            "url": "---"
        }],
        "categories": [item for k, item in categories.items()],
        "images": coco_images,
        "annotations": coco_annots
    }

    with open(output_file, "w") as f:
        json.dump(out, f)


def im_info(id, filename, size):
    return {
        "id": id,
        "width": size[1],
        "height": size[0],
        "file_name": filename,
        "license": 1,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": "",
    }


def annot_info(id, img_id, cat_id, segm, area, bbox):
    return {
        "id": id,
        "image_id": img_id,
        "category_id": cat_id,
        "segmentation": segm,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0
    }


if __name__ == '__main__':
    from glob import glob

    train_val_histos = {
        "001": "NA",
        "002": "NA",
        "003": "AD",
        "004": "AD",
        "005": "AD",
        "006": "AD",
        "007": "NA",
        "008": "NA",
        "009": "AD",
        "010": "AD",
        "011": "AD",
        "012": "NA",
        "013": "AD",
        "014": "AD",
        "015": "AD",
        "016": "AD",
        "017": "NA",
        "018": "AD",
    }

    test_histos = {
        "001": "AD",
        "002": "NA",
        "003": "AD",
        "004": "AD",
        "005": "AD",
        "006": "AD",
        "007": "AD",
        "008": "AD",
        "009": "AD",
        "010": "AD",
        "011": "AD",
        "012": "AD",
        "013": "AD",
        "014": "AD",
        "015": "AD",
        "016": "NA",
        "017": "AD",
        "018": "AD",
    }

    coco_info = {
        "description": "CVC-clinic",
        "url": "",
        "version": "1.0",
        "year": 2019,
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    categories = {
        "AD": {
            'id': 1,
            'name': 'AD',
            'supercategory': 'polyp',
        },
        "NA": {
            'id': 2,
            'name': 'NA',
            'supercategory': 'polyp',
        }
    }

    """
    folders_to_coco("/home/yael/PycharmProjects/detectron2/datasets/CVC-VideoClinicDBtrain_valid/images_train/",
                    "png",
                    "/home/yael/PycharmProjects/detectron2/datasets/CVC-VideoClinicDBtrain_valid/masks/",
                    "train.json",
                    train_val_histos)

    folders_to_coco("/home/yael/PycharmProjects/detectron2/datasets/CVC-VideoClinicDBtrain_valid/images_val/",
                    "png",
                    "/home/yael/PycharmProjects/detectron2/datasets/CVC-VideoClinicDBtrain_valid/masks/",
                    "valid.json",
                    train_val_histos)

    folders_to_coco("/home/yael/PycharmProjects/detectron2/datasets/cvcvideoclinicdbtest/images/",
                    "png",
                    "/home/yael/PycharmProjects/detectron2/datasets/cvcvideoclinicdbtest/masks/",
                    "test.json",
                    test_histos)
    """

