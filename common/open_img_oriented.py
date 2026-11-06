

#! /usr/bin/env python

import os
import PIL
from PIL import Image, ExifTags
import numpy as np
import django

# A human-confirmed rotation correction (ImageFile.rotation_scan_state/
# proposed_rotation_degrees -- see face_manager's rotation-review feature)
# is always one of these three discrete values -- the measured angle is
# snapped to whichever is closest before ever being applied as a real
# transform. Lives here (not in face_manager) because this is the single
# shared image-decode chokepoint every consumer already goes through for
# EXIF orientation, and the confirmed correction is applied the same way,
# right after it.
CANONICAL_ROTATIONS_DEG = (90.0, -90.0, 180.0)

# How far a degrees value may stray from an exact multiple of 90 (and
# still be accepted as one) -- see is_cardinal_rotation()'s own
# docstring. Generous enough to absorb ordinary floating-point/JS
# rounding on a value a frontend built from repeated +/-90 button
# presses, nowhere near loose enough to accept a genuinely different
# angle (e.g. 45 or 123.4) by mistake.
CARDINAL_ROTATION_TOLERANCE_DEGREES = 1.0


def circular_distance(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def snap_to_canonical_rotation(degrees):
    return min(CANONICAL_ROTATIONS_DEG, key=lambda c: circular_distance(degrees, c))


def is_cardinal_rotation(degrees):
    """True only for a degrees value that is (within
    CARDINAL_ROTATION_TOLERANCE_DEGREES) a NONZERO multiple of 90 -- 90,
    180, 270, -90, -180, -270, 450, etc. all count; 0, 360, -360 (no real
    rotation) and any non-90-multiple angle (e.g. 45, 123.4) do not.

    This is the only shape a human's rotate-left/right UI (built from
    repeated +/-90-degree button presses, per the frontend contract) can
    ever actually produce -- a confirm action that commits a human-chosen
    rotation must reject anything else outright rather than silently
    snapping an arbitrary value to whichever canonical bucket happens to
    be nearest (see snap_to_canonical_rotation's own tie-break risk right
    at 0). The audit's OWN measured proposals don't go through this check
    at all -- they're never anywhere near a multiple of 90 by construction
    (a real photo's residual misorientation is essentially never an exact
    quarter turn) and are snapped, not validated, when actually applied.
    """
    remainder = degrees % 90
    if min(remainder, 90 - remainder) > CARDINAL_ROTATION_TOLERANCE_DEGREES:
        return False
    mod360 = degrees % 360
    is_no_rotation = min(mod360, 360 - mod360) <= CARDINAL_ROTATION_TOLERANCE_DEGREES
    return not is_no_rotation


def _pil_transpose_op(snapped_degrees):
    return {90.0: Image.ROTATE_90, -90.0: Image.ROTATE_270, 180.0: Image.ROTATE_180}[snapped_degrees]


def apply_additional_rotation(image, degrees):
    """Applies a human-confirmed rotation correction ON TOP OF whatever
    EXIF orientation has already been applied -- these are two
    independent corrections composed in the order they were actually
    measured in, not the same correction applied twice: the confirmed
    angle was measured from face landmarks detected on the ALREADY-EXIF-
    corrected image (every detection pass already decodes through this
    same module), so it is by construction the residual rotation EXIF
    correction did not fix (e.g. a camera with no orientation sensor).

    `degrees` of None (no confirmed correction -- the overwhelmingly
    common case) returns `image` unchanged.
    """
    if degrees is None:
        return image
    return image.transpose(_pil_transpose_op(snap_to_canonical_rotation(degrees)))


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


def open_img_oriented(filename: str, as_numpy: bool, extra_rotation_degrees=None):
    # Open an image, get its metadata from the EXIF tag,
    # orient it, and then return as a numpy array.
    #
    # extra_rotation_degrees: a human-confirmed rotation correction (see
    # apply_additional_rotation's own docstring) -- applied AFTER EXIF
    # orientation, if given. The caller passes this explicitly (typically
    # ImageFile.confirmed_rotation_override_degrees) rather than this
    # function looking it up itself, since this is a generic, storage-
    # agnostic decode helper with no inherent DB dependency.
    
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

    # Keep a handle to the ORIGINALLY opened image specifically so its
    # underlying file can be force-closed in the `finally` below --
    # convert()/transpose() reassign `image` to independent derived
    # objects, so closing/loading this original reference afterward can't
    # affect whatever ends up being returned.
    original_image = image

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

        image = apply_additional_rotation(image, extra_rotation_degrees)

        if as_numpy:
            image = np.array(image)
            assert len(image.shape) == 3
    except OSError as e:
        print("EX", e)
        return None
    finally:
        # Pillow's own docs: Image.load() "will close the file associated
        # with the image" once decoded (unless draft() was used, which
        # nothing here does) -- but reading EXIF via _getexif() doesn't
        # necessarily trigger that, and neither convert() nor transpose()
        # guarantee the ORIGINAL file object gets closed just because the
        # local `image` variable was reassigned to a new object derived
        # from it. Left as-is, a tight loop calling this function over
        # many images (e.g. FaceExtractor.reencode_missing_faces()) leaks
        # one file descriptor per call -- confirmed in production
        # 2026-09-03: a 1,219-face batch hit "Too many open files" (ulimit
        # -n 1024) partway through, making 206 perfectly good images fail
        # with a misleading decode error. Forcing load() here explicitly
        # guarantees the close, is a harmless no-op if the file was
        # already fully loaded some other way, and never touches whatever
        # derived object is actually being returned.
        try:
            original_image.load()
        except Exception:
            pass
    return image

