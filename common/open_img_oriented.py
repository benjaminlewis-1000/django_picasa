

#! /usr/bin/env python

import os
import PIL
from PIL import Image, ExifTags
import numpy as np
import django

def apply_exif_orientation(image, orientation):
    """Apply the EXIF Orientation transform to a PIL Image, returning the
    corrected image. Handles all 8 standard values (1-8) via transpose(),
    which is the single source of truth going forward -- this used to be
    duplicated (and, in this module specifically, incompletely
    implemented -- only 3, 6, 8 via rotate(), silently doing nothing for
    2, 4, 5, 7) across common/open_img_oriented.py and
    filepopulator/models.py's ImageFile._init_image().

    orientation 0 is not a standard EXIF value, but is observed in
    practice (~1090 images in the live library) -- treated as equivalent
    to 1 (no rotation needed), same as "no EXIF orientation tag at all".
    Any other unrecognized value is also left untouched rather than
    guessed at.
    """
    if orientation in (0, 1):
        return image
    if orientation == 2:
        return image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    if orientation == 3:
        return image.transpose(PIL.Image.ROTATE_180)
    if orientation == 4:
        return image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    if orientation == 5:
        return image.transpose(PIL.Image.FLIP_LEFT_RIGHT).transpose(PIL.Image.ROTATE_90)
    if orientation == 6:
        return image.transpose(PIL.Image.ROTATE_270)
    if orientation == 7:
        return image.transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.ROTATE_90)
    if orientation == 8:
        return image.transpose(PIL.Image.ROTATE_90)
    return image


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
            image = apply_exif_orientation(image, exif[orientation])

        if as_numpy:
            image = np.array(image)
            assert len(image.shape) == 3
    except OSError as e:
        print("EX", e)
        return None
    return image

