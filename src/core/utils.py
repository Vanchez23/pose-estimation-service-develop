# -*- coding: utf-8 -*-
import cv2


def load_image(img_path):
    """
    Просто загружаем модель с помощью OpenCV.
    """
    cv2.IMREAD_COLOR
    image = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if image is None:
        raise Exception('Image was not loaded')
    return image
