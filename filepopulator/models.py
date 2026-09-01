from django.db import models
from django.conf import settings
from django.forms import ModelForm
from django.core.validators import MinValueValidator, MaxValueValidator, RegexValidator
from django.core.exceptions import ValidationError
from django.contrib.postgres.fields import ArrayField
# from django.core.validators import *
import re
from datetime import datetime
from django.utils import timezone
from rest_framework.reverse import reverse

from io import BytesIO
from django.core.files.base import ContentFile
import os
import time
from django.core.files.uploadedfile import SimpleUploadedFile
import hashlib
import PIL
from PIL.ExifTags import TAGS, GPSTAGS
from PIL import Image
from datetime import datetime
from django.core.files import File
import pytz
import logging
import cv2
import numpy as np
from fractions import Fraction
from dateutil import parser
import imagehash
import common
from PIL import ExifTags

# Accepted image file extensions -- shared by the filename validator here,
# process_new_no_md5()'s own check, and scripts.py's create_image_file()/
# add_from_root_dir() gates, so there's one place to extend when a new
# format is supported.
IMAGE_EXTENSION_REGEX = r"\.(?:[jJ][pP][eE]?[gG]|[hH][eE][iI][cC]|[hH][eE][iI][fF])$"

HEIC_EXTENSIONS = ('.heic', '.heif')

# Patterns for guess_date_from_filename(), each capturing (year, month,
# day, hour, minute, second). Order doesn't matter for correctness --
# guess_date_from_filename() tries all of them and picks the earliest
# plausible match, not just the first pattern to hit -- see its own
# docstring for why that matters.
_FILENAME_DATETIME_PATTERNS = [
    re.compile(r'(\d{4})-(\d{2})-(\d{2})[ _](\d{2})\.(\d{2})\.(\d{2})'),  # 2019-04-08 21.08.22
    re.compile(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})'),           # IMG_20240719_211850
    re.compile(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})'),       # 2016-02-06_16-18-35
]
# 13-digit millisecond Unix epoch (common in messaging-app media exports).
# Bounded to a "1[5-8]" leading pattern -- roughly 2017 through 2035 --
# purely to avoid matching an arbitrary 13-digit number that happens to
# sit in a filename (e.g. a database id) as if it were a timestamp; the
# real plausibility check is the MIN/MAX bound applied to every candidate
# below regardless of which pattern found it.
_FILENAME_EPOCH_MS_PATTERN = re.compile(r'(?<!\d)(1[5-8]\d{11})(?!\d)')

_FILENAME_DATE_MIN = pytz.utc.localize(datetime(1990, 1, 1))


def guess_date_from_filename(filename):
    """Best-effort date extraction from a filename, used ONLY as a
    fallback when EXIF metadata has no valid DateTime* field (see
    ImageFile._get_date_taken()) -- never overrides a real EXIF date.

    Tries several real-world naming conventions found in this library's
    own files (Android IMG_YYYYMMDD_HHMMSS, "YYYY-MM-DD HH.MM.SS",
    millisecond Unix epoch, etc.) and returns the EARLIEST plausible
    match found, not just the first pattern to match. That matters for
    filenames like a "Resized_" export that embed BOTH an original
    capture timestamp and a later re-export timestamp, sometimes in two
    different formats -- picking the first pattern-priority hit can
    silently return the later (wrong) date instead of the real one.

    Returns a tz-aware UTC datetime, or None if nothing plausible is
    found (roughly 37% of files with invalid EXIF in this library, per a
    2026-09 survey -- these are left exactly as before: no worse, not
    fixed either).
    """
    base = os.path.basename(filename)
    now = timezone.now()
    candidates = []

    for pattern in _FILENAME_DATETIME_PATTERNS:
        for m in pattern.finditer(base):
            y, mo, d, h, mi, s = (int(g) for g in m.groups())
            try:
                dt = pytz.utc.localize(datetime(y, mo, d, h, mi, s))
            except ValueError:
                continue  # e.g. month 13, day 32 -- not a real date, just digits
            if _FILENAME_DATE_MIN <= dt <= now:
                candidates.append(dt)

    for m in _FILENAME_EPOCH_MS_PATTERN.finditer(base):
        ms = int(m.group(1))
        try:
            dt = datetime.fromtimestamp(ms / 1000.0, tz=pytz.utc)
        except (ValueError, OSError, OverflowError):
            continue
        if _FILENAME_DATE_MIN <= dt <= now:
            candidates.append(dt)

    if not candidates:
        return None
    return min(candidates)


def _heic_style_exif(image):
    """Build an EXIF dict shaped like the legacy `Image._getexif()` API
    (decoded top-level tag names, GPSInfo as a raw numeric-keyed sub-dict)
    from the modern `Image.getexif()`/`get_ifd()` API.

    HEIC's pillow_heif-registered plugin doesn't implement `_getexif()` at
    all (confirmed: raises AttributeError) -- but everything downstream of
    building this dict (Make/Model/GPS/Orientation extraction, a few lines
    below in _init_image()) is otherwise format-agnostic, so this adapter
    lets HEIC reuse that same logic unchanged rather than duplicating it.
    """
    exif = image.getexif()
    if not exif:
        return None
    info = {}
    for tag_id, value in exif.items():
        name = TAGS.get(tag_id, tag_id)
        info[name] = value
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if gps_ifd:
        info['GPSInfo'] = dict(gps_ifd)
    return info

# Image thumbnail processing

# Create your models here.

# What do I want to happen when the images have the same hash? 
# What if the image changes? 
# What if the image moves? 
# -- Update the record.

# print(settings.LOG_LEVEL)
# logging.basicConfig(filename=settings.LOG_FILE, level=settings.LOG_LEVEL)


# logging.basicConfig(filename='example.log',level=logging.DEBUG)
settings.LOGGER.warning("TODO: Need to handle thumbnails in this better")
settings.LOGGER.warning("TODO: May want to handle bad files more gracefully.")
settings.LOGGER.warning("TODO: More logic is needed to see if the file has changed at all, instead of just hash")

# Latitude and longitude validators that accomodate out-of-GPS coordinates for
# default "not there" values.
def validate_lat(value):
    if value != -999:
        if not -90 < value < 90:
            raise ValidationError(f'{value} is not a valid latitude')

def validate_lon(value):
    if value != -999:
        if not -180 < value < 180:
            raise ValidationError(f'{value} is not a valid latitude')

def phash_to_bigint(image_hash):
    """Converts an imagehash.ImageHash (a 64-bit unsigned value) to the
    signed bigint Postgres/BigIntegerField actually stores -- same bit
    pattern numpy's int64 uses under the hood, so comparison code
    (filepopulator/similarity.py) can load this column straight into a
    numpy int64 array with no conversion."""
    unsigned = int(str(image_hash), 16)
    return int(np.uint64(unsigned).astype(np.int64))

class Directory(models.Model): 
    dir_path = models.CharField(max_length=512, unique=True)
    mean_datetime = models.DateTimeField(default = timezone.now)
    mean_datesec = models.FloatField(default = -1)
    first_datetime = models.DateTimeField(default = timezone.now)
    first_datesec = models.FloatField(default = -1)
    num_images = models.IntegerField(default=0)

    def __str__(self):
        return "{} --- ({})".format(self.dir_path.split('/')[-1], self.dir_path)

    def top_level_name(self):
        return "{}".format(self.dir_path.split('/')[-1])

#    def imgs_in_dir(self):
#        imgs = ImageFile.objects.filter(directory__dir_path=self.dir_path)
#        imgs = [i.id for i in imgs]
#        return imgs

    def __get_filtered_img_dates__(self, require_valid=True):
        imgs = ImageFile.objects.filter(directory__dir_path=self.dir_path)

        if require_valid:
            img_date = [time.mktime(i.dateTaken.timetuple()) for i in imgs if i.dateTakenValid]
        else:
            img_date = [time.mktime(i.dateTaken.timetuple()) for i in imgs if i.dateTaken is not None]
        img_date = np.array(img_date)

        def reject_outliers(data, m = 2.):
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            s = d/mdev if mdev else 0.
            return data[s<m]

        img_date =reject_outliers(img_date)
        if len(img_date) == 0:
            return None
        else:
            return img_date.reshape(-1)

    def __get_best_available_img_dates__(self):
        # Prefer EXIF-confirmed dates; if none of this directory's images
        # have a valid one (e.g. no EXIF DateTimeOriginal, so dateTaken
        # fell back to file-processing time), fall back to using whatever
        # dateTaken values exist anyway rather than defaulting to "now" --
        # a stale/unreliable date is still far more useful than the date
        # this happened to be recomputed on. Only when a directory has no
        # images with any dateTaken at all does this return None.
        img_dates = self.__get_filtered_img_dates__(require_valid=True)
        if img_dates is not None:
            return img_dates
        return self.__get_filtered_img_dates__(require_valid=False)

    def average_date_taken(self):
        img_dates = self.__get_best_available_img_dates__()
        if img_dates is None:
            self.mean_datesec = time.mktime(datetime.now().timetuple())
            self.mean_datetime = datetime.fromtimestamp(self.mean_datesec, pytz.utc)
        else:
            self.mean_datesec = float(np.mean(img_dates))
            self.mean_datetime = datetime.fromtimestamp(self.mean_datesec, pytz.utc)

    def beginning_date_taken(self):
        img_dates = self.__get_best_available_img_dates__()
        if img_dates is None:
            self.first_datesec = time.mktime(datetime.now().timetuple())
            self.first_datetime = datetime.fromtimestamp(self.first_datesec, pytz.utc)
        else:
            img_dates.sort()
            first_date = img_dates[0]
            self.first_datesec = int(first_date)
            self.first_datetime = datetime.fromtimestamp(self.first_datesec, pytz.utc)
    
def thumbnail_big_path(instance, filename):
    first_dir = filename[:2]
    second_dir = filename[2]
    return f"thumbnails_big/{first_dir}/{second_dir}/{filename}"

def thumbnail_med_path(instance, filename):
    first_dir = filename[:2]
    second_dir = filename[2]
    return f"thumbnails_med/{first_dir}/{second_dir}/{filename}"

def thumbnail_small_path(instance, filename):
    first_dir = filename[:2]
    second_dir = filename[2]
    return f"thumbnails_small/{first_dir}/{second_dir}/{filename}"

class DuplicateFile(models.Model):
    filename = models.CharField(max_length=1024)


class FailedImageFile(models.Model):
    # Tracks a file that has never successfully been ingested into
    # ImageFile -- e.g. corrupted/truncated from the moment it appeared in
    # the photo tree. A real ImageFile row can't be created for these:
    # ImageFile.save() requires a successful decode to populate
    # width/height/thumbnails. A previously-good ImageFile that later
    # becomes corrupted is tracked differently -- via its own
    # image_load_failed/image_load_error fields, since a full row already
    # exists for it. A future frontend view is planned to list files from
    # both sources for cleanup (see CLAUDE.md's Planned work).
    filename = models.CharField(max_length=1024, unique=True)
    error_message = models.TextField()
    # os.path.getctime() at the time of the failed attempt -- lets
    # add_from_root_dir() tell "still the same broken file, don't retry
    # every run" apart from "file was replaced/fixed, retry it".
    file_mod_time = models.FloatField()
    first_failed_at = models.DateTimeField(auto_now_add=True)
    last_attempted_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"FailedImageFile({self.filename})"


class GeocodeCache(models.Model):
    # Keyed by coordinate (rounded to ~11m -- see ROUND_DECIMALS below),
    # not by ImageFile: a photo library has massive location reuse (home,
    # relatives' houses, the same vacation spot across many shots), so
    # caching per-coordinate rather than per-image is what keeps a
    # Nominatim reverse-geocoding backfill within its rate limit.
    ROUND_DECIMALS = 4

    lat = models.FloatField()
    lon = models.FloatField()

    # Precise reverse-geocode result (e.g. from Nominatim) -- the actual
    # place this coordinate is in, however small/unrecognizable.
    locality = models.CharField(max_length=256, null=True, blank=True)
    county = models.CharField(max_length=256, null=True, blank=True)
    state = models.CharField(max_length=256, null=True, blank=True)
    country = models.CharField(max_length=256, null=True, blank=True)
    display_name = models.CharField(max_length=512, null=True, blank=True)
    raw_response = models.JSONField(null=True, blank=True)
    geocoded_at = models.DateTimeField(null=True, blank=True)
    lookup_failed = models.BooleanField(default=False)
    lookup_error = models.TextField(null=True, blank=True)

    # Nearest sufficiently-large place, independent of the precise lookup
    # above -- computed offline from a static dataset, no rate limit.
    # Lets downstream callers show either the precise locality or this
    # more-recognizable fallback (e.g. "Bothell" vs. "Seattle").
    nearest_metro_name = models.CharField(max_length=256, null=True, blank=True)
    nearest_metro_distance_km = models.FloatField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['lat', 'lon'], name='unique_geocode_cache_coordinate')
        ]

    def __str__(self):
        return f"GeocodeCache({self.lat}, {self.lon}) -> {self.display_name}"


# Lots ripped from https://github.com/hooram/ownphotos/blob/dev/api/models.py
class ImageFile(models.Model):

    filename = models.CharField(max_length=1024, validators=[RegexValidator(regex=IMAGE_EXTENSION_REGEX, message="Filename must be a JPG, JPEG, HEIC, or HEIF")], db_index = True)
    # CASCADE is expected; if delete directory, delete images.
    directory = models.ForeignKey(Directory, on_delete=models.PROTECT, related_name='image_set')
    pixel_hash = models.CharField(max_length = 64, null = False, default = -1)
    file_hash = models.CharField(max_length = 64, null = False, default = -1)

    full_res_path = models.CharField(max_length=1024, null=True, default=None)

    # Thumbnails 
    # thumbnail_big = models.ImageField(upload_to='thumbnails_big', editable=False, default=None)
    # thumbnail_medium = models.ImageField(upload_to='thumbnails_med', editable=False, default=None)
    # thumbnail_small = models.ImageField(upload_to='thumbnails_small', editable=False, default=None)
    thumbnail_big = models.ImageField(upload_to=thumbnail_big_path, editable=False, default=None)
    thumbnail_medium = models.ImageField(upload_to=thumbnail_med_path, editable=False, default=None)
    thumbnail_small = models.ImageField(upload_to=thumbnail_small_path, editable=False, default=None)

    # square_thumbnail = models.ImageField(upload_to='square_thumbnails')
    # square_thumbnail_tiny = models.ImageField(
    #     upload_to='square_thumbnails_tiny')
    # square_thumbnail_small = models.ImageField(
    #     upload_to='square_thumbnails_small')
    # square_thumbnail_big = models.ImageField(upload_to='square_thumbnails_big')

    # Fields for metadata
    camera_make = models.CharField(max_length = 64, null=True, blank=True)
    camera_model = models.CharField(max_length = 64, null=True, blank=True)
    flash_info = models.IntegerField(validators=[MinValueValidator(-1)], default= -1)
    exposure_num = models.IntegerField(validators=[MinValueValidator(-1)], default= -1)
    exposure_denom = models.IntegerField(validators=[MinValueValidator(-1)], default= -1)
    focal_num = models.IntegerField(validators=[MinValueValidator(-1)], default= -1)
    focal_denom = models.IntegerField(validators=[MinValueValidator(-1)], default= -1)
    fnumber_num = models.IntegerField(validators=[MinValueValidator(-1)], default= -1)
    fnumber_denom = models.IntegerField(validators=[MinValueValidator(-1)], default= -1)
    iso_value = models.IntegerField(validators=[MinValueValidator(-1)], default= -1)
    light_source = models.IntegerField(validators=[MinValueValidator(-1)], default= -1)
    gps_lat_decimal = models.FloatField(default=-999,validators=[validate_lat])
    gps_lon_decimal = models.FloatField(default=-999,validators=[validate_lon])

    # Where gps_lat_decimal/gps_lon_decimal actually came from. 'exif' for
    # everything ingested so far; room for a future personal
    # location-history service to backfill a position (with its own name
    # here) for images with no EXIF GPS, while keeping that provenance
    # visible since a location-history match is inherently less precise
    # than an EXIF GPS tag.
    gps_source = models.CharField(max_length=64, null=True, blank=True, default='exif')

    geocode = models.ForeignKey(
        GeocodeCache, on_delete=models.SET_NULL, null=True, blank=True, related_name='images'
    )

    # Default for date added is now.
    dateAdded = models.DateTimeField( default=timezone.now )
    dateModified = models.DateTimeField(default = timezone.now )
    width = models.IntegerField(validators=[MinValueValidator(1)])
    height = models.IntegerField(validators=[MinValueValidator(1)])

    # Default for date take in January 1, 1899.
    dateTaken = models.DateTimeField( default=datetime(2018, 1, 1) )
    dateTakenUTC = models.FloatField(default=0)
    dateTakenValid = models.BooleanField(default=False)

    # isProcessed -- whether the photo has had faces detected.
    isProcessed = models.BooleanField(default=False)
    orientation = models.IntegerField(default=-8008)

    # Set when the underlying file failed to open/decode -- e.g. a
    # corrupted or truncated JPEG. Generic (not tied to face extraction
    # specifically) since both face_manager's extraction pipeline and
    # filepopulator's own ingestion can hit this on the same file; a
    # future frontend view is planned to list these for cleanup.
    image_load_failed = models.BooleanField(default=False)
    image_load_error = models.TextField(null=True, blank=True)

    # 64-bit perceptual hash (imagehash.phash), stored as a signed bigint
    # -- the same bit pattern numpy's int64 uses, so comparison code can
    # load this column straight into a numpy array with no conversion.
    # Null until computed (e.g. a corrupted image that failed to decode).
    phash = models.BigIntegerField(null=True, blank=True, db_index=True)

    # Whether this image has been compared against the rest of the
    # library for near-duplicates (see filepopulator/similarity.py).
    # Comparison only ever needs to happen once per image: it's compared
    # against every other image that already has a phash at the time, and
    # every future image will in turn compare itself against this one
    # when its own turn comes -- same incremental-catch-up shape as
    # GeocodeCache's coordinate backfill.
    similarity_checked = models.BooleanField(default=False, db_index=True)

    # For storing tags
    tags = ArrayField(
        models.CharField(max_length = 128, null=True, blank = True),
        blank=True, null=True
    )

    description = models.CharField(max_length = 1024, null=True, blank = True)

    # thumbnail = models.ImageField(upload_to = settings.FILEPOPULATOR_THUMBNAIL_DIR, default = str(timezone.now) + '_thumbnail.jpg' )

    # def _get_full_path(self):
    #     expand_dir = self.directory.dir_path
    #     fullname = os.path.join(expand_dir, self.filename)
    #     return fullname


    def __str__(self):
        return "{}".format(self.filename)

    def process_new_no_md5(self):

        if not re.match(r".*" + IMAGE_EXTENSION_REGEX, self.filename):
            settings.LOGGER.debug("File {} does not have a supported image-type ending.".format(self.filename))
            return False # Success value

        self._init_image()
        self._get_dir()
        s = time.time()
        self._get_date_taken()

        return True # Success value

        # name_match = ImageFile.objects.filter(filename=self.filename)
        # if name_match: # i.e. we've looked at this file before.
        #     is_new = False
        #     assert len(name_match) == 1, 'More than one record for same image filepath.'
        #     old_pixel_hash = name_match[0].pixel_hash
        #     if old_pixel_hash == self.pixel_hash:
        #         # Still the same picture. We're good.
        #         return
        #     else:
        #         settings.LOGGER.debug("TODO: File repeat with changes. Need logic to see if the file has changed at all...")
                # raise NotImplementedError('What do we do here?')
            # return


        # self._generate_thumbnail()

        # hash_match = ImageFile.objects.filter(pixel_hash__contains=self.pixel_hash)
        # if hash_match and name_match:
        #     raise NotImplementedError('Hashes match! Need better info...')
        #     return
            # Do some other checks. Do the filenames match?
            # return
        # match = ImageFile.objects.filter(filename__contains=self.filename)# .get(image_hash = self.image_hash))
        # print(match.values('image_hash'))


    def _get_dir(self):
        directory_of_file = os.path.dirname( os.path.normpath( self.filename ) )

        try:
            self.directory = Directory.objects.get (dir_path = directory_of_file)
        except :
            instance = Directory(dir_path = directory_of_file)
            try:
                instance.full_clean()
            except ValidationError as ve:
                print(ve)
            else:
                instance.save()

            self.directory = Directory.objects.get(dir_path = directory_of_file)


    def _get_mod_time(self):
        self.dateModified = datetime.fromtimestamp(os.path.getctime(self.filename))

    def _init_image(self):
        # # Get the date taken:
        # # Source for these EXIF tag attributes: 
        # # https://www.awaresystems.be/imaging/tiff/tifftags/privateifd/exif.html
        # # exifDict = {}
        # # exifDict['DateTimeOriginal'] key = 36867
        # # exifDict['DateTimeDigitized'] key = 36868
        
        # # Source for this code snippet: 
        # # https://www.blog.pythonlibrary.org/2010/03/28/getting-photo-metadata-exif-using-python/
        # # Uses PIL to get a named dictionary of EXIF metadata.

        self.full_res_path = 'https://' + os.environ['MEDIA_DOMAIN'] + '/full_res' + self.filename.replace(settings.PHOTO_ROOT, '').replace(' ', '%20')

        s = time.time()
        try:
            self.image = PIL.Image.open(self.filename)
        except PIL.Image.DecompressionBombError:
            self.image = PIL.Image.fromarray(cv2.imread(self.filename))
        self._get_mod_time()
        # self.dateModified = datetime.fromtimestamp(os.path.getctime(self.filename))
        if self.dateModified.tzinfo == None:
            self.dateModified = self.dateModified.astimezone(pytz.utc)


        self.exifDict = {}
        is_heic = self.filename.lower().endswith(HEIC_EXTENSIONS)
        # print(self.filename)
        try:
            if is_heic:
                info = _heic_style_exif(self.image)
            else:
                info = self.image._getexif()
        except AttributeError as ae:
            info = None
        if info is not None:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                self.exifDict[decoded] = value
        else:
            self.orientation = 1
            self.exifDict = None

        if 'keys' in dir(self.exifDict):
            tags = ['Make', 'Model', 'Flash', 'ExposureTime', 'FocalLength', 'ISOSpeedRatings', 'FNumber', 'LightSource']

            if 'GPSInfo' in self.exifDict.keys():

                gps_keys = list(self.exifDict['GPSInfo'].keys())
                for key in gps_keys:
                    name = GPSTAGS.get(key,key)
                    self.exifDict['GPSInfo'][name] = self.exifDict['GPSInfo'].pop(key)
                # For *whatever* reason, some of the keys don't convert on the first pass.
                gps_keys = list(self.exifDict['GPSInfo'].keys())
                for key in gps_keys:
                    name = GPSTAGS.get(key,key)
                    self.exifDict['GPSInfo'][name] = self.exifDict['GPSInfo'].pop(key)

                def get_decimal_coordinates(info):
                    
                    for key in ['Latitude', 'Longitude']:
                        if 'GPS'+key in info and 'GPS'+key+'Ref' in info:
                            e = info['GPS'+key]
                            ref = info['GPS'+key+'Ref']
                            info[key] = ( e[0] +
                                          e[1] / 60 +
                                          e[2] / 3600
                                        ) * (-1 if ref in ['S','W'] else 1)

                    if 'Latitude' in info and 'Longitude' in info:
                        # print(info)
                        return [info['Latitude'], info['Longitude']]
                    else:
                        return [-999, -999]

                gps_lat, gps_lon = get_decimal_coordinates(self.exifDict['GPSInfo'])
                self.exifDict['GPSInfo']['GPSLatDec'] = gps_lat
                self.exifDict['GPSInfo']['GPSLonDec'] = gps_lon


            # We can - and should - decode the flash and light source 
            # values elsewhere, rather than putting that logic here. Plus 
            # I'm not feeling it right now. 
            def to_num_den(float_val, limit=5000):
                try:
                    frac = Fraction(float_val).limit_denominator(limit)
                except:
                    logging.debug(f"Exception! {float_val}, {limit}, {Fraction(float_val)}")
                    frac = Fraction(float_val)
                
                return frac.numerator, frac.denominator

            if 'Make' in self.exifDict.keys():
                make = self.exifDict['Make']
#                self.camera_make = make
                self.camera_make = make.replace('\x00', '').rstrip().lstrip()
            if 'Model' in self.exifDict.keys():
                model = self.exifDict['Model']
#                self.camera_model = model
                self.camera_model = model.replace('\x00', '').rstrip().lstrip()
            if 'Flash' in self.exifDict.keys():
                flash = self.exifDict['Flash']
                self.flash_info = flash
            if 'ExposureTime' in self.exifDict.keys():
                exposureTime = self.exifDict['ExposureTime']
                self.exposure_num, self.exposure_denom = to_num_den(exposureTime)
            if 'FocalLength' in self.exifDict.keys():
                focalLength = self.exifDict['FocalLength']
                self.focal_num, self.focal_denom = to_num_den(focalLength, 1000)
            if 'ISOSpeedRatings' in self.exifDict.keys():
                iso = self.exifDict['ISOSpeedRatings']
                self.iso_value = iso
            if 'FNumber' in self.exifDict.keys():
                fnumber = self.exifDict['FNumber']
                self.fnumber_num, self.fnumber_denom = to_num_den(fnumber, 100)
            if 'LightSource' in self.exifDict.keys():
                light_source = self.exifDict['LightSource']
                self.light_source = light_source
            if 'GPSInfo' in self.exifDict.keys():

                self.gps_lat_decimal = self.exifDict['GPSInfo']['GPSLatDec']
                if type(self.gps_lat_decimal) == Fraction:
                    self.gps_lat_decimal = float(self.gps_lat_decimal)
                if np.isnan(self.gps_lat_decimal):
                    # Bug fix: this used to fall back to 0 instead of the
                    # -999 "no GPS" sentinel used everywhere else (see
                    # get_decimal_coordinates() above, which does this
                    # correctly) -- (0, 0) is a real, valid-looking
                    # coordinate (off the coast of Africa), so a malformed
                    # GPS fraction that produced NaN was silently recorded
                    # as if the photo was actually taken there.
                    self.gps_lat_decimal = -999

                self.gps_lon_decimal = self.exifDict['GPSInfo']['GPSLonDec']
                if type(self.gps_lon_decimal) == Fraction:
                    self.gps_lon_decimal = float(self.gps_lon_decimal)
                if np.isnan(self.gps_lon_decimal):
                    self.gps_lon_decimal = -999
                    
            if 'Orientation' in self.exifDict.keys():
                self.orientation = self.exifDict['Orientation']
            else:
                self.orientation = 1

        if is_heic:
            # Empirically, pillow_heif/libheif auto-applies any
            # container-level rotation transform (irot/imir boxes) during
            # decode and resets the EXIF Orientation tag to 1 to match --
            # verified against 8 real-world iPhone HEIC samples (models 12
            # through 17 Pro), all of which came back as orientation 1
            # regardless of the photo's actual portrait/landscape framing.
            # A different value means either an encoder that behaves
            # differently than what's been tested, or something else
            # unexpected -- rather than guess at a second rotation on top
            # of whatever the decoder already did (risking a silently
            # wrong image), fail loudly. Raising a plain OSError here
            # routes through the same corrupted-image handling as
            # everything else (FailedImageFile / image_load_failed,
            # logged, not retried forever) via process_new_no_md5()'s
            # callers in scripts.py.
            if self.orientation != 1:
                msg = (
                    f"HEIC file {self.filename} has unexpected EXIF "
                    f"orientation {self.orientation} (expected 1) -- needs "
                    f"manual review before this can be trusted."
                )
                print(msg)
                settings.LOGGER.error(msg)
                raise OSError(msg)

            n_frames = getattr(self.image, 'n_frames', 1)
            if n_frames > 1:
                msg = (
                    f"HEIC file {self.filename} has {n_frames} frames "
                    f"(Live Photo or burst?) -- only single-frame HEIC is "
                    f"currently supported."
                )
                print(msg)
                settings.LOGGER.error(msg)
                raise OSError(msg)

        self.dateAdded = timezone.now()


        # Rotate depending on orientation. Shared with common/
        # open_img_oriented.py's apply_exif_orientation() -- this used to
        # be its own separate (and correct) implementation of the same
        # 8-value transform; now both call the one shared function.
        try:
            self.image = common.apply_exif_orientation(self.image, self.orientation)
        except:
            # Orientation 1
            pass

        self.width, self.height = self.image.size

    # Orientation ? 
    def _generate_md5_hash(self):
        # Reads the pixels in the image, reshapes them,
        # and then hash the pixels one by one using md5. 
        pixel_hash_md5 = hashlib.md5()

        try:
            self.pixels = cv2.cvtColor(np.array(self.image), cv2.COLOR_BGR2RGB)
        except TypeError as te:
            self.pixels = cv2.imread(self.filename)
        except PIL.Image.DecompressionBombError as bomberror:
            self.pixels = cv2.imread(self.filename)
        except OSError as oe:
            # A corrupted/truncated JPEG -- np.array(self.image) is where
            # PIL's lazy decode actually happens and raises. Fall back to
            # cv2.imread() the same way the other except branches above do;
            # it sometimes succeeds where PIL doesn't. If it can't either,
            # raise a clear, callers-can-catch-this OSError instead of the
            # cryptic AttributeError that self.pixels.reshape(-1) below
            # would otherwise raise on a None result.
            self.pixels = cv2.imread(self.filename)
            if self.pixels is None:
                raise OSError(f"Could not decode image pixels for {self.filename}: {oe}") from oe

        arr = self.pixels.reshape(-1)
        # arr = arr[::500]
        # arr = np.ascontiguousarray(arr)
        # print(arr)

        # Sample 1000 pixels in the array equally across the array.
        # This is deterministic. 
        # for idx in range(0,len(arr),max(len(arr)//1000, 1) ):
        #     it = arr[idx]
        #     pixel_hash_md5.update(bytes([it]))
        pixel_hash_md5.update(arr)
        
        self.pixel_hash = pixel_hash_md5.hexdigest()
        settings.LOGGER.debug(f'{self.pixel_hash}, {self.filename}')

        # Perceptual hash for near-duplicate detection (see
        # filepopulator/similarity.py). Computed here, not in a separate
        # pass, so it reuses self.image while it's still the full
        # decoded/oriented image -- _generate_thumbnail() (called right
        # after this in save()) resizes self.image in place. Best-effort:
        # a decode that got this far via the cv2.imread() fallback above
        # may still fail here (self.image can be an unusable lazy-opened
        # PIL handle on a truncated file); leave phash null rather than
        # let it take down the whole save().
        try:
            self.phash = phash_to_bigint(imagehash.phash(self.image))
        except Exception as e:
            settings.LOGGER.debug(f"Could not compute phash for {self.filename}: {e}")
            self.phash = None

        hash_file = hashlib.md5()
        # with open(self.filename, "rb") as f:
        #     for chunk in iter(lambda: f.read(4096), b""):
        #         hash_file.update(chunk)
        hash_file.update(self.filename.encode('utf-8'))
        self.file_hash = hash_file.hexdigest() 

        # Look for other files with same hashes
        other_hashed = ImageFile.objects.filter(pixel_hash = self.pixel_hash)
        if len(other_hashed):
            for obj in other_hashed:
                if os.path.exists(obj.filename):
                    pass
                    settings.LOGGER.debug("Assumption made about same pixel hash and existing file, but not same filename: do nothing.")
                   #  raise NotImplementedError('Same object hash -- what to do?')
                else:
                    pass
                    # Do nothing
                    settings.LOGGER.debug('Same hash as a deleted item. Watch this.')
                    # obj.thumbnail.delete()
                    # obj.delete()

    def _get_date_taken(self):
        # Default comparison date - we want the earliest date.
        self.dateTaken = timezone.now()
        # Flag for if we found a date taken in the EXIF data. 
        self.dateTakenValid = False

        if self.exifDict is not None:
            dateTakenKeys = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
            for exifKey in dateTakenKeys:
                datetaken_tmp = self.exifDict[exifKey] if exifKey in self.exifDict.keys() else None
                # Remediations for occasional problems - I've seen \x00\x00... in the string 
                # and date lines that are just spaces.
                if datetaken_tmp is None or re.match(r'^\s+$', datetaken_tmp) or re.match(r'0000:00:00 00:00:00', datetaken_tmp) or re.match(r'[\s:-]+', datetaken_tmp):
                    continue  # No value at this EXIF key
                else:
                    datetaken_tmp = datetaken_tmp.replace('\x00', '')
                    try:
                        date = datetime.strptime(datetaken_tmp, '%Y:%m:%d %H:%M:%S')
                    except ValueError as ve:
                        # settings.LOGGER.warning(f"Date taken format is _tmp}")
                        date = parser.parse(datetaken_tmp)
#                         date = datetime.strptime(datetaken_tmp, '%Y-%m-%d %H:%M:%S')
                    
                    if date.tzinfo == None:
                        date = pytz.utc.localize(date)
                    if date < self.dateTaken: 
                        self.dateTaken = date
                        self.dateTakenValid = True
                        self.dateTakenUTC = date.timestamp()
        else:
        #    self.dateTaken = timezone.now()
        #    self.dateTakenValid = False
            settings.LOGGER.warning(f"Date taken is not valid for file {self.filename}")

        if not self.dateTakenValid:
            # No real EXIF date -- try to salvage something better than
            # the now() placeholder above from the filename itself.
            # dateTakenValid stays False either way: this is a guess, not
            # EXIF-grade confidence, and nothing downstream that checks
            # dateTakenValid should start trusting it as if it were.
            filename_guess = guess_date_from_filename(self.filename)
            if filename_guess is not None:
                self.dateTaken = filename_guess

        # settings.LOGGER.error('Hi, debug here:')

        # Make the taken date timezone aware to get rid of warnings.
        # self.dateTaken = (self.dateTaken)

    def _generate_thumbnail(self):
        """
        Create and save the thumbnail for the photo (simple resize with PIL).
        """
        # fh = storage.open(self.photo_name, 'r')
        # try:
        #     image = Image.open(self.filename)
        # except:
        #     print("Couldn't open")  
        #     return False

        thumb_fields = [self.thumbnail_big, self.thumbnail_medium, self.thumbnail_small]
        thumb_sizes = [settings.FILEPOPULATOR_THUMBNAIL_SIZE_BIG, \
                        settings.FILEPOPULATOR_THUMBNAIL_SIZE_MEDIUM, \
                        settings.FILEPOPULATOR_THUMBNAIL_SIZE_SMALL]

        # JPEG (FTYPE below) can't encode an alpha channel or a palette --
        # found via a real production HEIC file that decoded to RGBA mode
        # (some HEIC images carry an alpha plane even for plain photos,
        # unlike the all-RGB samples this was originally tested against),
        # which raised "cannot write mode RGBA as JPEG" here. Converting
        # drops any alpha channel; for a real photo, that channel is
        # essentially always fully opaque anyway.
        if self.image.mode not in ('RGB', 'L'):
            self.image = self.image.convert('RGB')

        for field, size in zip(thumb_fields, thumb_sizes):


            image = self.image

            image.thumbnail(size, Image.LANCZOS)

#            thumb_dir = self.pixel_hash[:2]
            thumb_filename = f'{self.pixel_hash}_{self.file_hash}.jpg'

            FTYPE = 'JPEG' # 'GIF' or 'PNG' are possible extensions

            # Save thumbnail to in-memory file as StringIO
            temp_thumb = BytesIO()
            image.save(temp_thumb, FTYPE)
            temp_thumb.seek(0)

            # Load a ContentFile into the thumbnail field so it gets saved
            field.save(thumb_filename, ContentFile(temp_thumb.read()), save=False)

            temp_thumb.close()

        return True

    def save(self, *args, **kwargs):
        """
        Make and save the thumbnail for the photo here.
        """
        # I only do the MD5 hash in the save function because it
        # is so expensive. I also have to redo the _init_image function
        # for some reason, so that the self.image field is populated
        # appropriately (it somehow loses it...)

        self._init_image()
        self._generate_md5_hash()

        if not self._generate_thumbnail():
            raise Exception('Could not create thumbnail - is the file type valid?')
        super(ImageFile, self).save(*args, **kwargs)

    def delete(self):
        # Face.source_image_file references this ImageFile with
        # on_delete=CASCADE. Django's cascade-delete collector removes
        # those Face rows with a bulk SQL DELETE, which does NOT call each
        # Face's overridden delete() (the one that removes its
        # face_thumbnail file from disk) -- so those thumbnail files were
        # being silently orphaned on disk every time an ImageFile was
        # deleted this way (e.g. delete_removed_photos(), run on every
        # scheduled ingestion pass for photos that vanished from disk).
        # Import here, not at module level, to avoid a circular import --
        # face_manager/models.py already imports filepopulator.
        from face_manager.models import Face
        for face in Face.objects.filter(source_image_file=self):
            face.delete()

        # file = ImageFile.objects.filter(id=self.id)
        # os.remove(file[0].thumbnail_small.path)
        try:
            os.remove(self.thumbnail_big.path)
        except Exception as e: 
            pass
        try:
            os.remove(self.thumbnail_medium.path)
        except Exception as e: 
            pass
        try:
            os.remove(self.thumbnail_small.path)
        except Exception as e: 
            pass
        super(ImageFile, self).delete()


    # def admin_thumbnail(self):
    #         func = getattr(self, 'get_admin_thumbnail_url', None)
    #         if func is None:
    #             return _('An "admin_thumbnail" photo size has not been defined.')
    #         else:
    #             if hasattr(self, 'get_absolute_url'):
    #                 return mark_safe(u'<a href="{}"><img src="{}"></a>'.format(self.get_absolute_url(), func()))
    #             else:
    #                 return mark_safe(u'<a href="{}"><img src="{}"></a>'.format(self.image.url, func()))

    # admin_thumbnail.short_description = ('Thumbnail')
    # admin_thumbnail.allow_tags = True
    def image_img(self):
        if self.image:
            return marksafe('<img src="%s" />' % self.thumbnail_small.url)
        else:
            return '(Sin imagen)'
        image_img.short_description = 'Thumb'


    def exposure(self):
        return f"{self.exposure_num}/{self.exposure_denom}"

    exposure.short_description = 'Exposure'


class SimilarImagePair(models.Model):
    """One edge in the near-duplicate graph: two ImageFiles whose phash
    Hamming distance was <= settings.PHASH_SIMILARITY_THRESHOLD the last
    time either was compared. image_a/image_b are always stored with
    image_a_id < image_b_id (see record()) so the same pair can't be
    inserted twice as (A, B) and (B, A) -- the incremental comparison
    task compares every image against the full population each run, so
    without this a pair would otherwise get rediscovered (and hit the
    unique constraint or duplicate) from both directions.

    CASCADE (not the sentinel-reassignment pattern Face uses) is correct
    here: this is just a graph edge with no side effects of its own, so
    deleting either endpoint should simply drop the edge.
    """
    image_a = models.ForeignKey(ImageFile, on_delete=models.CASCADE, related_name='similar_to_higher_id')
    image_b = models.ForeignKey(ImageFile, on_delete=models.CASCADE, related_name='similar_to_lower_id')
    hamming_distance = models.PositiveSmallIntegerField()
    discovered_at = models.DateTimeField(default=timezone.now)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['image_a', 'image_b'], name='unique_similar_image_pair')
        ]

    def __str__(self):
        return f"SimilarImagePair({self.image_a_id}, {self.image_b_id}, dist={self.hamming_distance})"

    @classmethod
    def record(cls, image_id_a, image_id_b, distance):
        lo, hi = sorted((image_id_a, image_id_b))
        cls.objects.update_or_create(
            image_a_id=lo, image_b_id=hi,
            defaults={'hamming_distance': distance},
        )


