
#! /usr/bin/env python

import os
import re
import PIL
from PIL import Image
from PIL.ExifTags import TAGS 
import cv2
from datetime import datetime

project_path = os.path.abspath(os.path.join(__file__,"../.."))
script_path  = os.path.abspath(os.path.join(__file__,".."))

print('Todo: Config file, e.g. for thumbnail sizes')

class metadata_extractor():
    """docstring for JPEG_metadata"""
    def __init__(self, filename):
        # Set some basic values like the filename
        self.filename = filename

        # Assert that it's a file that exists and that it
        # ends in JPEG type extension. 
        assert os.path.exists(self.filename)
        assert re.match(".*\.[j|J][p|P][e|E]?[g|G]$", self.filename), 'Filename must end in some variant of "JPEG".'
    
        # Get the directory
        self.file_dir = os.path.dirname( os.path.normpath( self.filename) )

        # Read the image for both the Exif data and the pixels. 
        self.PIL_img = Image.open(self.filename)
        self.img_pixels = cv2.imread(self.filename)

        # Get shape info
        self.shape = self.img_pixels.shape[:2]
        self.height = self.shape[0]
        self.width = self.shape[1]

        # Define constants. Thumbnail sizes will be defined in a config file. 
        self.cw_action = 'clockwise'
        self.ccw_action = 'ccw'
        self.r180_action = 'r_180'

        self.big_thumb_size = 400
        self.mid_thumb_size = 200
        self.small_thumb_size = 100


        self._get_exif_dict_()
        self._hash_img()
        self._get_date_taken_()
        self._make_thumbnails_()

        # Other stuff that I may want to stuff in to a method.
        self.orientation = self.exifDict['Orientation']



    def detect_matching_class(self, other_metadata_class):
        # Function to compare with other images. Requires an instance
        # of this class for the other image. Tests to see if the 
        # hashes are equal and if the length and width dimensions 
        # are equal (regardless of orientation)
        assert isinstance(other_metadata_class, metadata_extractor)

        if self.img_hash_val == other_metadata_class.img_hash_val \
            and set(self.shape) == set(other_metadata_class.shape):
            return True
        else:
            return False

    def detect_matching_values(self, other_hash, other_size):
        # Function to compare with other images. Requires the hash
        # for the other image and the image size. Tests to see if the 
        # hashes are equal and if the length and width dimensions 
        # are equal (regardless of orientation)
        if self.img_hash_val == other_hash and set(self.shape) == set(other_size):
            return True
        else:
            return False

    def _get_exif_dict_(self):
        # Get the exif values from PIL and write them to a dictionary. 
        self.exifDict = {}
        info = self.PIL_img._getexif()
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            self.exifDict[decoded] = value

    def __rotate_cv_img__(self, rotate_command):
        # Helper function to rotate an OpenCV image. Found it on the internet somewhere,
        # but I have tested and know it works. 
        if rotate_command == self.ccw_action:
            altered_img = cv2.transpose(self.img_pixels)
            altered_img = cv2.flip(altered_img, flipCode=0)
        elif rotate_command == self.cw_action:
            altered_img = cv2.transpose(self.img_pixels)
            altered_img = cv2.flip(altered_img, flipCode=1)
        elif rotate_command == self.r180_action:
            altered_img = cv2.flip(self.img_pixels, flipCode=0)
            altered_img = cv2.flip(altered_img, flipCode=1)
        else:
            raise ValueError('Rotate command passed was not valid')
            self.stop_event.set()

        return altered_img

    def _hash_img(self):
        # Find the hash of the image and all three other rotations.
        # The hash of the image is the minimum of all these hashes.
        # Useful for determining if an image is the same but rotated. 

        # Hash the original
        hash_orig = hash(self.img_pixels.tostring())
        # Hash clockwise
        hash_cw = hash(
            self.__rotate_cv_img__(self.cw_action).tostring()
            )
        # Hash counter-clockwise
        hash_ccw = hash(
            self.__rotate_cv_img__(self.ccw_action).tostring()
            )
        # Hash rotated by 180
        hash_180 = hash(
            self.__rotate_cv_img__(self.r180_action).tostring()
            )

        # To detect simple rotations, we want the minimum hash. It cuts
        # our hash space somewhat, but collisions are still unlikely. 
        minhash = min(hash_orig, hash_ccw, hash_cw, hash_180)
        self.img_hash_val = minhash

    def _get_date_taken_(self):
        # Default comparison date - we want the earliest date from the 
        # exif data. We'll start with the dateTaken as now, and then
        # see if we can find anything earlier. 
        self.dateTaken = datetime.now()
        # Flag for if we found a date taken in the EXIF data. 
        self.dateTakenValid = False

        # Regex search for any Exif key with 'date' in it. 
        dateSearch = re.compile('.*date.*', re.IGNORECASE)
        # dateTakenKeys = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
        dateTakenKeys = list(filter(dateSearch.match, self.exifDict.keys() ) )

        for exifKey in dateTakenKeys:
            datetaken_tmp = self.exifDict[exifKey]
            if datetaken_tmp is None:
                continue  # No value at this EXIF key
            else:
                # Check to see if this key has an earlier date than
                # our current data. 
                date = datetime.strptime(datetaken_tmp, '%Y:%m:%d %H:%M:%S')
                if date < self.dateTaken: 
                    self.dateTaken = date
                    self.dateTakenValid = True

    def _make_thumbnails_(self):
        # Function to make smaller thumbnails of the image.
        # If the desired size is smaller than we currently have,
        # then the thumbnail is just the image. 
        min_dimension = float(min(self.shape))
        max_dimension = float(max(self.shape))

        if max_dimension / min_dimension > 2:
            raise ValueError('Aspect ratio of this image may need special handling when making thumbnail.')

        # I'm aware that this may need more work, if the aspect ratio is crazy. 

        def __make_thumbnail__(thumb_size):
            ratio = thumb_size / min_dimension
            if ratio < 1:
                out_thumb = cv2.resize(self.img_pixels, dsize=(0, 0), fx = ratio, fy = ratio)
            else:
                out_thumb = self.img_pixels

            return out_thumb

        self.big_thumb = __make_thumbnail__(self.big_thumb_size)  
        self.mid_thumb = __make_thumbnail__(self.mid_thumb_size)  
        self.small_thumb = __make_thumbnail__(self.small_thumb_size)  



if __name__ == '__main__':
    filename = os.path.join(project_path, 'test_imgs', 'orientation', '2_1.jpg' )
    f1 = metadata_extractor(filename)
    print(f1.img_pixels.shape)

    filename = os.path.join(project_path, 'test_imgs', 'orientation', '2_8.jpg' )
    f2 = metadata_extractor(filename)
    print(f2.img_pixels.shape)

    print f1.detect_matching_class(f2)
    print f1.detect_matching_values(f2.img_hash_val, f2.shape)
