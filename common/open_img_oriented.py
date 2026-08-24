

#! /usr/bin/env python

import os
import PIL
from PIL import Image, ExifTags
import numpy as np
import django

def open_img_oriented(filename: str, as_numpy: bool):
    # Open an image, get its metadata from the EXIF tag,
    # orient it, and then return as a numpy array
    
    if type(filename) == django.db.models.fields.files.ImageFieldFile:
#        data = filename.read()    
        image = PIL.Image.open(filename.file)
    elif type(filename) == str:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")

        try:
            image = PIL.Image.open(filename)
        except Exception as e:
            print("EX", e)
            return None
    else:
        raise NotImplementedError(f"Type of filename is {type(filename)}")

    # PIL.Image.open() parses the header lazily and succeeds even on a
    # truncated/broken file -- the real decode (and OSError) happens on
    # first actual pixel access, which any of convert()/rotate()/
    # np.array() below can trigger depending on exactly where the file is
    # damaged. Catch it around all of them, not just open() above, so this
    # function actually honors its documented "returns None on failure"
    # contract instead of raising partway through.
    try:
        if image.mode == 'L':
            # If a grayscale image, convert to 3-channel "RGB"
            image = image.convert('RGB')

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

        if as_numpy:
            image = np.array(image)
            assert len(image.shape) == 3
    except OSError as e:
        print("EX", e)
        return None
    return image

