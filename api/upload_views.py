#! /usr/bin/env python

"""User-facing file upload endpoint.

Accepts one image, video, or zip archive per POST request from an
authenticated user, validates it for real (content decode, not just
extension/Content-Type), and stages it under settings.UPLOAD_STAGING_DIR
for the existing filepopulator ingestion scan to pick up on its next
scheduled pass -- no separate ingestion path needed, since
UPLOAD_STAGING_DIR is a subdirectory of the already-scanned PHOTO_ROOT.

Design choices (see CLAUDE.md for the full discussion this session):
- One file per request, not a batch -- simpler retry/progress semantics
  for the frontend (a multi-file picker just fires one request per
  file).
- The client sends a `checksum` field (hex-encoded SHA-256 of the raw
  file bytes); computed server-side over the same bytes while they're
  being written to a scratch temp file (a single pass, not a second
  read) and compared before anything is trusted or moved into the real
  staging directory.
- The scratch temp file lives OUTSIDE the scanned photo tree entirely
  (Python's own tempfile default location), so the ingestion scanner can
  never see a partially-written or not-yet-validated file, regardless of
  timing.
- A zip's members are extracted into UPLOAD_STAGING_DIR individually,
  each going through the exact same content-verification as a directly
  uploaded file. Zip-slip is defeated by discarding any path component
  in a member's name (os.path.basename) before ever joining it to a real
  path, plus a realpath containment check as a second layer. A nested
  zip, or any non-image/video member, is skipped and reported rather
  than treated as a hard error -- per the user's own call that an
  authenticated-only endpoint doesn't need to defend against
  deliberately malicious zip content the way a public-facing one would;
  the real risks here are structural (path traversal, zip bombs), not
  code execution, since zipfile itself never executes archive content.
"""

import hashlib
import os
import re
import shutil
import tempfile
import uuid
import zipfile

import av
from django.conf import settings
from PIL import Image
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from filepopulator.models import IMAGE_EXTENSION_REGEX, VIDEO_EXTENSION_REGEX

_IMAGE_EXT_RE = re.compile(IMAGE_EXTENSION_REGEX)
_VIDEO_EXT_RE = re.compile(VIDEO_EXTENSION_REGEX)
_ZIP_EXT_RE = re.compile(r"\.[zZ][iI][pP]$")
_CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")


def _classify_extension(filename):
    """None means "not an accepted type" -- the caller reports that as a
    rejection/skip rather than guessing."""
    if _IMAGE_EXT_RE.search(filename):
        return 'image'
    if _VIDEO_EXT_RE.search(filename):
        return 'video'
    if _ZIP_EXT_RE.search(filename):
        return 'zip'
    return None


def _verify_image(path):
    """Real content verification, not just a header sniff -- a renamed
    text file or a truncated transfer fails one of these two steps.
    verify() alone doesn't catch everything (it deliberately avoids
    fully decoding pixel data), so a second real open+load is needed
    too -- PIL's own documented reason a fresh Image.open() is required
    after verify() rather than reusing the same handle."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
    except Exception:
        return False
    return True


def _verify_video(path):
    try:
        with av.open(path) as container:
            return len(container.streams.video) > 0
    except Exception:
        return False


def _verify_content(path, kind):
    if kind == 'image':
        return _verify_image(path)
    if kind == 'video':
        return _verify_video(path)
    return False


def _unique_destination(directory, basename):
    """Two different zip entries (from different subfolders inside the
    archive) can share the same basename once path components are
    stripped for zip-slip safety -- disambiguate rather than silently
    overwrite one with the other."""
    dest = os.path.join(directory, basename)
    if not os.path.exists(dest):
        return dest
    root, ext = os.path.splitext(basename)
    return os.path.join(directory, f"{root}_{uuid.uuid4().hex[:8]}{ext}")


class UploadFileView(APIView):
    """POST /api/upload/ -- multipart form with a single `file` field and
    a `checksum` field (hex-encoded SHA-256 of the file's raw bytes).

    Authenticated users only. IsAuthenticated is also this project's
    global DRF default (REST_FRAMEWORK['DEFAULT_PERMISSION_CLASSES']),
    set explicitly here anyway for the same reason the slideshow-facing
    views are explicit about their own permission class -- this endpoint
    must never be reachable via the slideshow key.
    """

    permission_classes = (IsAuthenticated,)
    parser_classes = (MultiPartParser,)

    def post(self, request):
        upload = request.FILES.get('file')
        if upload is None:
            return Response(
                {'error': 'No file provided (expected multipart field "file").'},
                status=400,
            )

        checksum = request.data.get('checksum', '').strip().lower()
        if not _CHECKSUM_RE.match(checksum):
            return Response(
                {'error': 'A hex-encoded SHA-256 "checksum" field is required.'},
                status=400,
            )

        kind = _classify_extension(upload.name)
        if kind is None:
            return Response({
                'error': f'Unsupported file type for "{upload.name}" -- '
                         f'expected an image, video, or .zip file.',
            }, status=400)

        if upload.size > settings.UPLOAD_MAX_FILE_SIZE_BYTES:
            return Response({
                'error': f'File too large ({upload.size} bytes, max '
                         f'{settings.UPLOAD_MAX_FILE_SIZE_BYTES}).',
            }, status=413)

        # Single pass: hash while writing to a scratch temp file outside
        # the scanned photo tree entirely, so the ingestion scanner can
        # never see a partially-written or not-yet-validated file.
        hasher = hashlib.sha256()
        fd, tmp_path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'wb') as tmp:
                for chunk in upload.chunks():
                    hasher.update(chunk)
                    tmp.write(chunk)

            if hasher.hexdigest() != checksum:
                return Response({
                    'error': 'Checksum mismatch -- upload may have been '
                             'corrupted in transit.',
                }, status=400)

            batch_dir = os.path.join(settings.UPLOAD_STAGING_DIR, uuid.uuid4().hex)

            if kind == 'zip':
                body, all_ok, any_ok = self._handle_zip(tmp_path, batch_dir)
                status = 201 if all_ok else (207 if any_ok else 400)
                return Response(body, status=status)

            if not _verify_content(tmp_path, kind):
                return Response({
                    'status': 'rejected',
                    'files': [{
                        'filename': upload.name, 'status': 'rejected',
                        'reason': f'not a valid {kind} file',
                    }],
                }, status=400)

            os.makedirs(batch_dir, exist_ok=True)
            dest = os.path.join(batch_dir, os.path.basename(upload.name))
            shutil.move(tmp_path, dest)
            return Response({
                'status': 'accepted',
                'files': [{'filename': upload.name, 'status': 'accepted'}],
            }, status=201)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _handle_zip(self, tmp_path, batch_dir):
        """Returns (response_body, all_accepted, any_accepted)."""
        results = []
        any_accepted = False
        any_rejected_or_skipped = False

        try:
            with zipfile.ZipFile(tmp_path) as zf:
                bad_entry = zf.testzip()
                if bad_entry is not None:
                    return (
                        {'status': 'rejected',
                         'error': f'Zip archive is corrupted (bad CRC for "{bad_entry}").'},
                        False, False,
                    )

                infolist = zf.infolist()
                if len(infolist) > settings.UPLOAD_MAX_ZIP_ENTRY_COUNT:
                    return (
                        {'status': 'rejected',
                         'error': f'Zip archive has too many entries '
                                  f'({len(infolist)}, max {settings.UPLOAD_MAX_ZIP_ENTRY_COUNT}).'},
                        False, False,
                    )
                total_uncompressed = sum(i.file_size for i in infolist)
                if total_uncompressed > settings.UPLOAD_MAX_ZIP_UNCOMPRESSED_BYTES:
                    return (
                        {'status': 'rejected',
                         'error': f'Zip archive is too large uncompressed '
                                  f'({total_uncompressed} bytes, max '
                                  f'{settings.UPLOAD_MAX_ZIP_UNCOMPRESSED_BYTES}).'},
                        False, False,
                    )

                for info in infolist:
                    if info.is_dir():
                        continue
                    basename = os.path.basename(info.filename)
                    if not basename:
                        # A traversal-only entry name (e.g. "../../x") with
                        # nothing real left after stripping path
                        # components -- nothing safe to extract to.
                        results.append({'filename': info.filename, 'status': 'skipped',
                                         'reason': 'unsafe or empty entry name'})
                        any_rejected_or_skipped = True
                        continue

                    entry_kind = _classify_extension(basename)
                    if entry_kind == 'zip':
                        results.append({'filename': info.filename, 'status': 'skipped',
                                         'reason': 'nested zip archives are not processed'})
                        any_rejected_or_skipped = True
                        continue
                    if entry_kind is None:
                        results.append({'filename': info.filename, 'status': 'skipped',
                                         'reason': 'unsupported file type'})
                        any_rejected_or_skipped = True
                        continue

                    os.makedirs(batch_dir, exist_ok=True)
                    dest = _unique_destination(batch_dir, basename)
                    # Defense in depth beyond the basename-only join above --
                    # confirm the resolved path still lands inside batch_dir.
                    if os.path.dirname(os.path.realpath(dest)) != os.path.realpath(batch_dir):
                        results.append({'filename': info.filename, 'status': 'rejected',
                                         'reason': 'unsafe path'})
                        any_rejected_or_skipped = True
                        continue

                    with zf.open(info) as src, open(dest, 'wb') as out:
                        shutil.copyfileobj(src, out)

                    if not _verify_content(dest, entry_kind):
                        os.remove(dest)
                        results.append({'filename': info.filename, 'status': 'rejected',
                                         'reason': f'not a valid {entry_kind} file'})
                        any_rejected_or_skipped = True
                        continue

                    results.append({'filename': info.filename, 'status': 'accepted'})
                    any_accepted = True
        except zipfile.BadZipFile:
            return ({'status': 'rejected', 'error': 'Not a valid zip archive.'}, False, False)
        except (RuntimeError, NotImplementedError) as e:
            # Encrypted archives (RuntimeError: "password required") or an
            # exotic/unsupported compression method -- real, expected
            # rejections, not a server error.
            return ({'status': 'rejected', 'error': f'Could not read zip archive: {e}'}, False, False)

        all_accepted = any_accepted and not any_rejected_or_skipped
        return ({'status': 'accepted' if any_accepted else 'rejected', 'files': results},
                all_accepted, any_accepted)
