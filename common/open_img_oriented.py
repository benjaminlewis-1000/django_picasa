

#! /usr/bin/env python

import os
import PIL
from PIL import Image, ExifTags
import numpy as np

def open_img_oriented(filename: str, as_numpy: bool):
    # Open an image, get its metadata from the EXIF tag,
    # orient it, and then return as a numpy array

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    try:
        image = PIL.Image.open(filename)
    except Exception as e:
        print("EX", e)
        return None

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
            break

    try:
        exif=dict(image._getexif().items())
    except Exception as e:
        exif = {}

    if orientation in exif.keys():
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)

    # print(image.shape)
    if as_numpy:
        image = np.array(image)
    return image

