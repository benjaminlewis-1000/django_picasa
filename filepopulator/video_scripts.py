#! /usr/bin/env python

"""Video ingestion -- parallel to scripts.py's image ingestion, not merged
into it. See CLAUDE.md's "Video support" write-up for the full design and
why VideoFile is a separate model/pipeline rather than folded into
ImageFile's.

Metadata is deliberately split across two tools rather than one:
ffprobe (stream/technical info -- width/height/duration/codec, and later
frame extraction) and exiftool (GPS/creation-date/camera-make-model/
rotation -- confirmed empirically to recover meaningfully more than
ffprobe alone, e.g. a real Sony camcorder .MTS file where ffprobe found
no creation_time at all but exiftool recovered DateTimeOriginal, Make,
and Model). exiftool's Rotation tag is used as the sole source of
rotation -- exiftool and ffprobe's own side_data rotation disagreed on
sign convention for the same real file, so mixing the two would be
actively wrong, not just redundant.
"""

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime

import common
import pytz
from django.conf import settings
from django.utils import timezone

from .models import Directory, FailedVideoFile, VideoFile, VIDEO_EXTENSION_REGEX, guess_date_from_filename

FFPROBE_TIMEOUT_SECONDS = 30
EXIFTOOL_TIMEOUT_SECONDS = 30

EXIFTOOL_FIELDS = [
    '-CreateDate', '-DateTimeOriginal',
    # '#' forces numeric (not human-readable-string) output for just
    # these two tags -- a global -n flag was tried first and rejected:
    # it also disables print-conversion for Make/Model, and at least
    # one real Sony camcorder file has an undecoded numeric Make tag
    # that -n then reports as a meaningless raw code (264) instead of
    # the correct decoded "Sony" string exiftool gives without -n.
    '-GPSLatitude#', '-GPSLongitude#',
    '-Rotation', '-Make', '-Model',
]


def _run_ffprobe(file_path):
    """Returns parsed ffprobe JSON (dict with 'format'/'streams'), or None
    if ffprobe itself fails or the output can't be parsed -- callers
    treat that as "this file couldn't be read as video at all"."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', file_path],
            capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        settings.LOGGER.error(f"ffprobe failed to run against {file_path}: {e}")
        return None

    if result.returncode != 0:
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _run_exiftool(file_path):
    """Returns exiftool's metadata dict for file_path, or {} if exiftool
    fails or finds nothing -- a real, permanent case for some old
    digitized home movies with no embedded metadata at all, not
    necessarily an error."""
    try:
        result = subprocess.run(
            ['exiftool', '-j'] + EXIFTOOL_FIELDS + [file_path],
            capture_output=True, text=True, timeout=EXIFTOOL_TIMEOUT_SECONDS,
        )
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError, IndexError):
        return {}
    return data[0] if data else {}


def _record_video_failure(file_path, error_message):
    mod_time = os.path.getctime(file_path) if os.path.exists(file_path) else 0
    FailedVideoFile.objects.update_or_create(
        filename=file_path,
        defaults={'error_message': error_message, 'file_mod_time': mod_time},
    )
    settings.LOGGER.error(f"Video file {file_path} failed to process: {error_message}")


def _parse_exif_date(date_str):
    """exiftool date strings look like '2025:11:27 01:31:30' (no tz) or
    '2018:12:14 20:15:14-04:00 DST' (tz + zone name).

    dateutil.parser is deliberately NOT used as a fallback here -- tried
    first, then rejected against a real file: it doesn't understand
    exiftool's 'YYYY:MM:DD' colon-separated date at all, and silently
    defaults the date portion to today() while still parsing the time
    correctly, with NO exception raised. Confirmed against a real Sony
    camcorder file: '2018:12:14 20:15:14-04:00' silently became
    today's date at 20:15:14-04:00 instead of failing loudly. Explicit
    strptime formats (including a %z-aware one for the offset case)
    avoid this entirely."""
    if not date_str:
        return None
    # Drop a trailing zone-name word (e.g. "DST") exiftool sometimes
    # appends after a numeric offset -- neither format below expects it.
    cleaned = date_str.rsplit(' ', 1)[0] if date_str[-1].isalpha() else date_str
    date = None
    for fmt in ('%Y:%m:%d %H:%M:%S%z', '%Y:%m:%d %H:%M:%S'):
        try:
            date = datetime.strptime(cleaned, fmt)
            break
        except ValueError:
            continue
    if date is None:
        return None
    if date.tzinfo is None:
        date = pytz.utc.localize(date)
    return date


def _get_video_date_taken(file_path, exif):
    """Same fallback order as ImageFile._get_date_taken(): a real
    metadata date first, then a best-effort guess from the filename
    (many of these video filenames embed a capture timestamp the same
    way phone photo filenames do), then now() as a last resort with
    dateTakenValid=False either way for the filename-guess and no-match
    cases."""
    date = _parse_exif_date(exif.get('DateTimeOriginal') or exif.get('CreateDate'))
    if date is not None:
        return date, True

    filename_guess = guess_date_from_filename(file_path)
    if filename_guess is not None:
        return filename_guess, False

    return timezone.now(), False


def create_video_file(file_path):
    """Create or update a VideoFile row for file_path. Mirrors
    create_image_file()'s shape but without pixel-hash-based duplicate
    detection (see VideoFile.file_hash's own docstring -- not worth the
    cost of decoding a whole video just to hash it) and without
    thumbnail/face-detection work, which are later phases."""

    probe = _run_ffprobe(file_path)
    if probe is None:
        _record_video_failure(file_path, "ffprobe could not read this file")
        return

    video_stream = next(
        (s for s in probe.get('streams', []) if s.get('codec_type') == 'video'), None
    )
    if video_stream is None:
        _record_video_failure(file_path, "No video stream found")
        return

    try:
        width = int(video_stream['width'])
        height = int(video_stream['height'])
    except (KeyError, TypeError, ValueError):
        _record_video_failure(file_path, "Could not determine video dimensions")
        return

    try:
        duration = float(probe.get('format', {}).get('duration', 0) or 0)
    except (TypeError, ValueError):
        duration = 0

    codec = video_stream.get('codec_name')

    exif = _run_exiftool(file_path)
    date_taken, date_taken_valid = _get_video_date_taken(file_path, exif)

    directory_path = os.path.dirname(file_path)
    directory, _ = Directory.objects.get_or_create(dir_path=directory_path)

    video = VideoFile.objects.filter(filename=file_path).first() or VideoFile(filename=file_path)
    video.directory = directory
    video.width = width
    video.height = height
    video.duration_seconds = duration
    video.codec = codec
    video.camera_make = exif.get('Make')
    video.camera_model = exif.get('Model')
    try:
        video.rotation = int(exif.get('Rotation') or 0)
    except (TypeError, ValueError):
        video.rotation = 0
    video.dateTaken = date_taken
    video.dateTakenUTC = date_taken.timestamp()
    video.dateTakenValid = date_taken_valid
    video.dateModified = datetime.fromtimestamp(os.path.getctime(file_path))

    lat = exif.get('GPSLatitude')
    lon = exif.get('GPSLongitude')
    if lat is not None and lon is not None:
        video.gps_lat_decimal = lat
        video.gps_lon_decimal = lon
        video.gps_source = 'exif'

    hash_file = hashlib.md5()
    hash_file.update(file_path.encode('utf-8'))
    video.file_hash = hash_file.hexdigest()

    video.save()

    # Clear a stale failure record now that this file has actually
    # succeeded (mirrors create_image_file()'s equivalent handling).
    FailedVideoFile.objects.filter(filename=file_path).delete()


def add_videos_from_root_dir(root_dirs):
    """Walk each directory in root_dirs (a list, not a single path --
    VIDEO_ROOTS is an explicit allowlist of subfolders under the one
    VIDEO_ROOT bind mount, plus PHOTO_ROOT for videos interspersed in
    the existing photo tree) and create/update a VideoFile row for any
    new or changed video file found.

    One advisory lock covers the whole call (all roots), not one per
    root -- matches add_from_root_dir()'s own locking, own distinct
    lock name so the two ingestion pipelines never contend with each
    other."""
    with common.advisory_lock('filepopulator.add_videos_from_root_dir') as acquired:
        if not acquired:
            print("Locked!")
            return

        actual_file_list = []
        for root_dir in root_dirs:
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.startswith('.'):
                        continue
                    if re.search(VIDEO_EXTENSION_REGEX, f):
                        actual_file_list.append(os.path.join(root, f))

        db_file_list = list(VideoFile.objects.values_list('filename', flat=True))

        unchanged_failed_file_list = [
            f.filename for f in FailedVideoFile.objects.all()
            if os.path.exists(f.filename) and os.path.getctime(f.filename) == f.file_mod_time
        ]

        new_files = list(
            set(actual_file_list) - set(db_file_list) - set(unchanged_failed_file_list)
        )
        print(f"New video file length is {len(new_files)}")

        for filename in new_files:
            try:
                create_video_file(filename)
            except Exception as e:
                print(f"{filename} was not processed. {e}")
                _record_video_failure(filename, str(e))
