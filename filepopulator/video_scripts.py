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
import time
from datetime import datetime

import common
import pytz
from django.conf import settings
from django.utils import timezone

from .models import (
    Directory, DuplicateVideoFile, FailedVideoFile, PhotoCollection, VideoFile, VIDEO_EXTENSION_REGEX,
    guess_date_from_filename,
)

FFPROBE_TIMEOUT_SECONDS = 30
EXIFTOOL_TIMEOUT_SECONDS = 30

# Chunked read size for _compute_content_hash() -- large enough to keep
# per-chunk Python overhead negligible, small enough not to matter for
# memory on even the biggest real files in this library (~3.5GB).
_CONTENT_HASH_CHUNK_BYTES = 8 * 1024 * 1024

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


def compute_video_content_hash(file_path):
    """Real MD5 of the file's actual bytes, not just its path (see
    VideoFile.content_hash's own docstring for the real-world duplicate
    case this exists to catch). Benchmarked 2026-09-22 against real
    production video files: ~130-138 MB/s -- ~70 minutes for the entire
    542GB library, confirmed cheap enough to run unconditionally on
    every newly-ingested video, not just as a one-time backfill.

    Unlike ImageFile.pixel_hash (an MD5 of decoded PIXEL data, verified
    against actual pixel arrays before trusting a match -- see
    _pixel_arrays_match()), this hashes the raw file bytes directly, so
    a match here already IS byte-for-byte file identity -- no separate
    content-verification step is needed the way pixel_hash needed one."""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(_CONTENT_HASH_CHUNK_BYTES), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_video_playback_path(file_path):
    """VideoFile.playback_path's computation, extracted so
    backfill_video_playback_path (for rows ingested before this field
    existed) can reuse the exact same logic rather than a second copy
    that could drift. Mirrors ImageFile._init_image()'s own
    full_res_path computation. PHOTO_ROOT-rooted paths (videos
    interspersed in the photo tree) use the existing /full_res Apache
    mount; everything else (the real VIDEO_ROOT library) uses the new
    /video_original mount -- VIDEO_ROOT itself was never reachable
    through Apache before Phase 4 (slideshow serving)."""
    if file_path.startswith(settings.PHOTO_ROOT):
        rel = file_path.replace(settings.PHOTO_ROOT, '')
        mount = '/full_res'
    else:
        rel = file_path.replace(settings.VIDEO_ROOT, '')
        mount = '/video_original'
    return 'https://' + os.environ['MEDIA_DOMAIN'] + mount + rel.replace(' ', '%20')


def create_video_file(file_path):
    """Create or update a VideoFile row for file_path. Mirrors
    create_image_file()'s shape, including real content-based duplicate
    detection (via compute_video_content_hash(), added 2026-09-22 --
    see VideoFile.content_hash's own docstring for why: real duplicate
    video files were found being independently face-extracted, each
    producing its own set of Face rows for the same real moment) --
    but still without thumbnail/face-detection work, which are later
    phases."""

    # If this exact path was previously recorded as a content-duplicate
    # of some other video, wipe that bookkeeping and start fresh --
    # mirrors create_image_file()'s own top-of-function guard. Without
    # this, a path whose real content later genuinely changes (the file
    # at this path was replaced) would just leave the old
    # DuplicateVideoFile record sitting there stale, pointing at
    # whatever used to be the "original" -- never wrong in a way that
    # loses data (a VideoFile row is never created at a path with a
    # live DuplicateVideoFile record, so there's nothing to clean up on
    # that side), but confusing bookkeeping left unresolved otherwise.
    stale_duplicates = DuplicateVideoFile.objects.filter(filename=file_path)
    if stale_duplicates.exists():
        # Per-instance delete, not a bulk queryset delete -- shouldn't
        # normally find anything here (a VideoFile row and a
        # DuplicateVideoFile record are mutually exclusive at the same
        # path), but if it ever does, VideoFile.delete()'s own override
        # (Face/thumbnail cleanup) needs to actually run.
        for stale_video in VideoFile.objects.filter(filename=file_path):
            stale_video.delete()
        for dupe in stale_duplicates:
            dupe.delete()
        return

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

    if duration < settings.MIN_VIDEO_DURATION_SECONDS:
        # Not a real failure -- a deliberate exclusion by design (Live
        # Photos/motion clips, accidental taps). Recorded via
        # FailedVideoFile anyway so a re-scan doesn't re-run ffprobe
        # against the same never-changing short file every time (same
        # mtime-check mechanism add_videos_from_root_dir() already uses
        # for real failures) -- checked before the exiftool call, so
        # excluding a file also skips that cost entirely.
        _record_video_failure(
            file_path,
            f"Excluded: duration {duration:.1f}s is below the "
            f"{settings.MIN_VIDEO_DURATION_SECONDS}s minimum",
        )
        return

    codec = video_stream.get('codec_name')

    existing_same_path = VideoFile.objects.filter(filename=file_path).first()
    content_hash = compute_video_content_hash(file_path)

    if existing_same_path is not None:
        # Re-ingesting the same path (e.g. its mtime changed) -- never a
        # "duplicate" of itself, just the normal update-in-place flow.
        video = existing_same_path
    else:
        # A genuinely new path -- check whether its content already
        # exists elsewhere in the library before creating anything.
        content_matches = list(
            VideoFile.objects.filter(content_hash=content_hash).exclude(content_hash__isnull=True)
        )
        still_present = [v for v in content_matches if os.path.exists(v.filename)]
        if still_present:
            # Real duplicate: identical bytes already ingested at
            # another, still-present path. Record and stop -- mirrors
            # create_image_file()'s pixel_hash duplicate branch,
            # including the bug that branch had to fix (never fall
            # through to also creating a full row for the duplicate).
            #
            # Also cross-links the collection this duplicate was found
            # under onto the primary's own `collections` -- otherwise
            # the same real video landing in two collections would
            # correctly avoid a second row, but have no record it's
            # actually visible from both (added 2026-10-21, mirrors
            # create_image_file()'s equivalent duplicate-collection
            # linking).
            dup_collection = PhotoCollection.resolve_for_path(file_path)
            DuplicateVideoFile.objects.create(
                filename=file_path, original=still_present[0], collection=dup_collection,
            )
            if dup_collection is not None:
                still_present[0].collections.add(dup_collection)
            settings.LOGGER.info(
                f"Video {file_path} is a content-duplicate of "
                f"{still_present[0].filename} -- recorded, not re-ingested."
            )
            return

        moved_candidates = [v for v in content_matches if not os.path.exists(v.filename)]
        if moved_candidates:
            # Same content, but the row's old path is gone -- this is a
            # move/rename, not a duplicate. Reuse the existing row (and
            # its Face rows/thumbnails) instead of creating a fresh one.
            video = moved_candidates[0]
            video.filename = file_path
        else:
            video = VideoFile(filename=file_path)

    exif = _run_exiftool(file_path)
    date_taken, date_taken_valid = _get_video_date_taken(file_path, exif)

    directory_path = os.path.dirname(file_path)
    # Mirrors ImageFile._get_dir() -- resolves which PhotoCollection this
    # path belongs to (longest-prefix match against every registered
    # root_directory, falling back to the default collection) the one
    # time this Directory row is first created. Added 2026-10-21: this
    # was previously a plain get_or_create with no collection resolution
    # at all, leaving every video Directory's collection permanently
    # NULL -- see backfill_video_directory_collection for the one-time
    # catch-up on rows created before this fix.
    directory, created = Directory.objects.get_or_create(dir_path=directory_path)
    if created:
        directory.collection = PhotoCollection.resolve_for_path(directory_path)
        directory.save()

    video.content_hash = content_hash
    video.directory = directory
    video.width = width
    video.height = height
    video.duration_seconds = duration
    video.codec = codec
    video.field_order = video_stream.get('field_order', 'unknown')
    video.color_transfer = video_stream.get('color_transfer', 'unknown')
    # Mirrors video_face_pipeline.py's ffprobe_info() own SAR-correction
    # computation (a separate ffprobe-parsing code path, same as
    # field_order/color_transfer above) -- 0 means "checked, square
    # pixels", distinct from the field's own NULL default ("not yet
    # checked").
    sar = video_stream.get('sample_aspect_ratio', '1:1')
    sar_num, _, sar_den = sar.partition(':')
    try:
        sar_ratio = float(sar_num) / float(sar_den) if sar_den and float(sar_den) != 0 else 1.0
    except ValueError:
        sar_ratio = 1.0
    if sar_ratio != 1.0:
        corrected = round(width * sar_ratio)
        video.sar_scale_width = corrected + (corrected % 2)
    else:
        video.sar_scale_width = 0
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

    try:
        video.file_size_bytes = os.path.getsize(file_path)
    except OSError:
        video.file_size_bytes = None

    video.playback_path = compute_video_playback_path(file_path)

    video.save()

    # Mirrors instance_clean_and_save()'s equivalent call for images --
    # a freshly-saved video's own home collection is always itself one
    # of the collections it's "visible from."
    if video.directory.collection_id is not None:
        video.collections.add(video.directory.collection_id)

    # Clear a stale failure record now that this file has actually
    # succeeded (mirrors create_image_file()'s equivalent handling).
    FailedVideoFile.objects.filter(filename=file_path).delete()


def add_videos_from_root_dir(root_dirs):
    """Walk each directory in root_dirs (a list, not a single path --
    the caller, filepopulator.tasks.load_videos_into_db(), builds this
    from every PhotoCollection's own root_directory plus its
    extra_video_roots, an explicit allowlist of dedicated video-only
    subfolders under the shared VIDEO_ROOT bind mount) and create/update
    a VideoFile row for any new or changed video file found.

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
            found_in_this_root = 0
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.startswith('.'):
                        continue
                    if re.search(VIDEO_EXTENSION_REGEX, f):
                        actual_file_list.append(os.path.join(root, f))
                        found_in_this_root += 1

            # Same reasoning as add_from_root_dir()'s own check (image
            # side): a root walking to zero files is routine for a
            # brand-new, genuinely-empty root, but exceptional if
            # VideoFile rows are already tracked under it -- a possible
            # dropped mount/unreachable source, not routine content churn.
            if found_in_this_root == 0 and VideoFile.objects.filter(
                filename__startswith=root_dir
            ).exists():
                message = (
                    f"Scan of {root_dir} found 0 video files, but VideoFile rows are already "
                    f"tracked under it -- possible unmounted/unreachable source."
                )
                settings.LOGGER.error(f"add_videos_from_root_dir: {message}")
                common.log_error('filepopulator.video_scripts.add_videos_from_root_dir', message)

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


def delete_removed_videos():
    """Mirrors scripts.py's delete_removed_photos() -- a VideoFile row
    for a file that's since vanished from disk (moved, renamed outside
    this app, deleted) isn't cleaned up by add_videos_from_root_dir()
    alone, which only ever adds/updates. Previously a deliberate Phase 1
    scope cut (see CLAUDE.md); built once a real case came up (a VTS_*
    DVD-rip .avi moved out of the library, 2026-09-15). Uses .delete()
    per-instance (not a bulk queryset .delete()) so VideoFile.delete()'s
    own override runs -- cleans up the video's thumbnails and its Face
    rows' thumbnails too, not just the DB rows.

    Doesn't delete a row the first time its file is found missing -- see
    VideoFile.missing_since's own comment and
    settings.MISSING_FILE_GRACE_PERIOD for why (the same real incident
    that motivated the image-side equivalent, 2026-10-05: a bulk move
    raced this, permanently losing rows -- and their tagged Face rows --
    before a later scan's own move-detection ever got a chance to run)."""
    now = timezone.now()
    for video in VideoFile.objects.all():
        if not os.path.isfile(video.filename):
            if video.missing_since is None:
                # .update(), not .save() -- matches the image-side
                # equivalent's own reasoning (avoid any save()-triggered
                # side effects on a row whose file doesn't exist), even
                # though VideoFile has no custom save() override today.
                VideoFile.objects.filter(pk=video.pk).update(missing_since=now)
            elif now - video.missing_since >= settings.MISSING_FILE_GRACE_PERIOD:
                video.delete()
        elif video.missing_since is not None:
            VideoFile.objects.filter(pk=video.pk).update(missing_since=None)


# Phase 5 (transcode pipeline), added 2026-10-15 -- targeting a
# Raspberry Pi viewer with no hardware HEVC decode, so HEVC is treated
# the same as any other non-web-native codec, not special-cased for
# Safari the way it would be for a general-browser audience.
TRANSCODE_TIMEOUT_SECONDS = 3600
# CRF 28, not the usual "visually near-lossless" 23 -- this is a casual
# slideshow viewing context, not something anyone's pixel-peeping, and
# CRF scales roughly logarithmically (every +6 or so roughly halves
# bitrate), so this is a real, substantial size win. preset slow (not
# medium) -- this is a one-time backlog, not a tight loop, so trading
# more encode time for meaningfully better compression at the same CRF
# is worth it. Both picked 2026-10-15; see CLAUDE.md for the real-file
# CRF 23 vs 28 comparison this was validated against.
TRANSCODE_CRF = 33
TRANSCODE_PRESET = 'slow'


def _sharded_transcode_path(content_hash):
    """Mirrors the thumbnail_*_path() sharding convention (filepopulator/
    models.py) -- avoids one huge flat directory. Keyed by content_hash,
    not VideoFile.id: the same real bytes always land at the same
    output path regardless of which row currently references them, so a
    move/re-ingest (create_video_file()'s own move-reuse branch) never
    needs to redo a transcode that already exists."""
    return os.path.join(content_hash[:2], content_hash[2], f'{content_hash}.mp4')


def transcode_video(video):
    """Transcodes one VideoFile to H.264/AAC/mp4 under
    settings.VIDEO_TRANSCODE_ROOT. Mutates the row via .update() (not
    .save(), matching this file's own established reasoning -- no
    custom save() override exists on VideoFile today, but this avoids
    ever needing to care). Never raises -- records transcode_failed/
    _error instead, same convention as face_extraction_failed/_error."""
    if not video.content_hash:
        VideoFile.objects.filter(pk=video.pk).update(
            transcode_failed=True, transcode_error="No content_hash computed yet.",
        )
        return

    rel_path = _sharded_transcode_path(video.content_hash)
    out_path = os.path.join(settings.VIDEO_TRANSCODE_ROOT, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Write to a .tmp sibling and rename into place at the end -- a
    # crash/kill mid-encode must never leave a half-written file at the
    # real path looking like a finished transcode.
    tmp_path = out_path + '.tmp'
    cmd = [
        'ffmpeg', '-y', '-i', video.filename,
        '-c:v', 'libx264', '-crf', str(TRANSCODE_CRF), '-preset', TRANSCODE_PRESET,
        '-vf', 'setsar=1',
        '-c:a', 'aac',
        '-movflags', '+faststart',
        # Real bug found via a live smoke test (2026-10-15): ffmpeg
        # infers the output container from the filename's extension by
        # default, and the .tmp suffix (needed for the atomic-rename
        # safety above) isn't a format it recognizes -- "Unable to
        # choose an output format". -f mp4 makes the format explicit
        # regardless of what the path's own extension looks like.
        '-f', 'mp4',
        tmp_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TRANSCODE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        VideoFile.objects.filter(pk=video.pk).update(
            transcode_failed=True,
            transcode_error=f"ffmpeg timed out after {TRANSCODE_TIMEOUT_SECONDS}s",
        )
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return

    if result.returncode != 0 or not os.path.isfile(tmp_path):
        VideoFile.objects.filter(pk=video.pk).update(
            transcode_failed=True, transcode_error=(result.stderr or '')[-2000:],
        )
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return

    os.replace(tmp_path, out_path)
    VideoFile.objects.filter(pk=video.pk).update(
        transcoded_path=rel_path, transcode_failed=False, transcode_error=None,
    )


def transcode_pending_videos(limit=None, max_runtime_seconds=None):
    """Transcodes every VideoFile whose codec isn't web-native and
    hasn't been transcoded (or already failed) yet. Deliberately simple
    -- one video at a time, serially -- since ffmpeg's own re-encode
    already saturates available CPU on its own (same reasoning this
    project already settled on for ONNX/numpy work: process-level
    parallelism just oversubscribes the same cores).

    max_runtime_seconds, if given, stops the loop between videos once
    elapsed wall time exceeds it -- same convention as
    face_manager.tasks.process_video_faces(), used to run this in
    bounded daily windows rather than one long pass."""
    qs = VideoFile.objects.exclude(codec__in=settings.WEB_NATIVE_CODECS).filter(
        transcoded_path__isnull=True, transcode_failed=False,
    ).order_by('id')
    if limit:
        qs = qs[:limit]

    done = failed = 0
    start_time = time.time()
    for video in qs.iterator(chunk_size=20):
        if max_runtime_seconds is not None and (time.time() - start_time) >= max_runtime_seconds:
            settings.LOGGER.debug(
                f"Video transcode: time budget ({max_runtime_seconds}s) reached, stopping between videos."
            )
            break
        transcode_video(video)
        if VideoFile.objects.filter(pk=video.pk, transcode_failed=True).exists():
            failed += 1
        else:
            done += 1
    return done, failed
