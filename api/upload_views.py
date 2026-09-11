#! /usr/bin/env python

"""User-facing file upload endpoint.

Accepts one image, video, or zip archive from an authenticated user,
validates it for real (content decode, not just extension/Content-Type),
and stages it under settings.UPLOAD_STAGING_DIR for the existing
filepopulator ingestion scan to pick up on its next scheduled pass -- no
separate ingestion path needed, since UPLOAD_STAGING_DIR is a
subdirectory of the already-scanned PHOTO_ROOT.

Two entry points share the same validation/staging logic
(_stage_validated_file):
- UploadFileView: a single request, whole file at once -- fine for
  ordinary photos and short clips.
- The chunked flow (InitChunkedUploadView / UploadChunkView /
  ChunkedUploadStatusView / CompleteChunkedUploadView): for anything
  large enough that a single request risks exceeding a reasonable
  timeout (a multi-GB video over a modest connection can take many
  minutes to transfer) or that benefits from being resumable after a
  dropped connection. See CLAUDE.md's 2026-09-10 write-up for the full
  design discussion.

Design choices:
- The client sends a `checksum` field (hex-encoded SHA-256 of the raw
  file bytes -- of each chunk for the chunked flow, AND of the whole
  reassembled file, checked again once every chunk is in); computed
  server-side over the same bytes while they're being written to a
  scratch temp file (a single pass, not a second read) and compared
  before anything is trusted or moved into the real staging directory.
- Scratch data (single-shot temp files, and chunked-upload pieces) lives
  OUTSIDE the scanned photo tree entirely, so the ingestion scanner can
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
import math
import os
import re
import shutil
import tempfile
import uuid
import zipfile

import av
from django.conf import settings
from django.shortcuts import get_object_or_404
from PIL import Image
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from api.models import UploadChunk, UploadSession
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
    """Two different files can end up wanting the same basename in the
    same flat directory -- two different zip entries from different
    subfolders once path components are stripped for zip-slip safety, or
    two separate uploads (at different times, possibly different users)
    of a same-named file, since UPLOAD_STAGING_DIR is intentionally flat
    (no per-upload subdirectory -- see CLAUDE.md's 2026-09-11 write-up).
    Disambiguate with a short random suffix rather than silently
    overwriting one with the other.

    Accepted, low-probability race: this checks-then-creates rather than
    atomically reserving the name, so two uploads of the identical
    filename landing at the exact same instant could both pick the same
    "doesn't exist yet" name. Not hardened against further -- this is a
    private multi-user family app, not a high-concurrency public one, and
    the cost of a real collision is just re-uploading the losing file."""
    dest = os.path.join(directory, basename)
    if not os.path.exists(dest):
        return dest
    root, ext = os.path.splitext(basename)
    return os.path.join(directory, f"{root}_{uuid.uuid4().hex[:8]}{ext}")


def _chown_upload_path(path):
    """Uploaded content should be owned by the real host user
    (settings.UPLOAD_FILE_OWNER_UID/GID), not root. The container runs
    as root (see CLAUDE.md), so anything it creates defaults to
    root:root ownership -- awkward for anything on the host side (Samba
    browsing, manual cleanup) that expects the same ownership as the
    rest of the photo tree. A no-op if either setting is unset (e.g.
    local dev, where the process already runs as the real user).
    Best-effort: never let an ownership hiccup block a real upload."""
    uid = getattr(settings, 'UPLOAD_FILE_OWNER_UID', None)
    gid = getattr(settings, 'UPLOAD_FILE_OWNER_GID', None)
    if uid is None or gid is None:
        return
    try:
        os.chown(path, uid, gid)
    except OSError:
        pass


def _handle_zip(tmp_path, dest_dir):
    """Returns (response_body, http_status). Shared by both the
    single-shot and chunked-complete entry points."""
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
                    400,
                )

            infolist = zf.infolist()
            if len(infolist) > settings.UPLOAD_MAX_ZIP_ENTRY_COUNT:
                return (
                    {'status': 'rejected',
                     'error': f'Zip archive has too many entries '
                              f'({len(infolist)}, max {settings.UPLOAD_MAX_ZIP_ENTRY_COUNT}).'},
                    400,
                )
            total_uncompressed = sum(i.file_size for i in infolist)
            if total_uncompressed > settings.UPLOAD_MAX_ZIP_UNCOMPRESSED_BYTES:
                return (
                    {'status': 'rejected',
                     'error': f'Zip archive is too large uncompressed '
                              f'({total_uncompressed} bytes, max '
                              f'{settings.UPLOAD_MAX_ZIP_UNCOMPRESSED_BYTES}).'},
                    400,
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

                os.makedirs(dest_dir, exist_ok=True)
                dest = _unique_destination(dest_dir, basename)
                # Defense in depth beyond the basename-only join above --
                # confirm the resolved path still lands inside dest_dir.
                if os.path.dirname(os.path.realpath(dest)) != os.path.realpath(dest_dir):
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

                _chown_upload_path(dest)
                results.append({'filename': info.filename, 'status': 'accepted'})
                any_accepted = True
    except zipfile.BadZipFile:
        return ({'status': 'rejected', 'error': 'Not a valid zip archive.'}, 400)
    except (RuntimeError, NotImplementedError) as e:
        # Encrypted archives (RuntimeError: "password required") or an
        # exotic/unsupported compression method -- real, expected
        # rejections, not a server error.
        return ({'status': 'rejected', 'error': f'Could not read zip archive: {e}'}, 400)

    all_accepted = any_accepted and not any_rejected_or_skipped
    status = 201 if all_accepted else (207 if any_accepted else 400)
    return ({'status': 'accepted' if any_accepted else 'rejected', 'files': results}, status)


def _stage_validated_file(tmp_path, original_filename):
    """Given a file already sitting at tmp_path (its checksum already
    verified by the caller), classify + content-verify + stage it under
    UPLOAD_STAGING_DIR. Returns (response_body, http_status). Shared by
    UploadFileView (single-shot) and CompleteChunkedUploadView.

    Staging is deliberately FLAT -- every accepted file (whether a
    direct upload or a zip member) lands directly in UPLOAD_STAGING_DIR
    itself, not a per-upload subdirectory (that was the original design;
    changed 2026-09-11 per the user's own preference for a flat layout).
    _unique_destination handles the resulting cross-upload collision
    case that the old per-upload subdirectory used to avoid for free."""
    kind = _classify_extension(original_filename)
    if kind is None:
        return ({'error': f'Unsupported file type for "{original_filename}" -- '
                           f'expected an image, video, or .zip file.'}, 400)

    os.makedirs(settings.UPLOAD_STAGING_DIR, exist_ok=True)
    _chown_upload_path(settings.UPLOAD_STAGING_DIR)

    if kind == 'zip':
        return _handle_zip(tmp_path, settings.UPLOAD_STAGING_DIR)

    if not _verify_content(tmp_path, kind):
        return ({
            'status': 'rejected',
            'files': [{
                'filename': original_filename, 'status': 'rejected',
                'reason': f'not a valid {kind} file',
            }],
        }, 400)

    dest = _unique_destination(settings.UPLOAD_STAGING_DIR, os.path.basename(original_filename))
    shutil.move(tmp_path, dest)
    _chown_upload_path(dest)
    return ({
        'status': 'accepted',
        'files': [{'filename': original_filename, 'status': 'accepted'}],
    }, 201)


class UploadFileView(APIView):
    """POST /api/upload/ -- multipart form with a single `file` field and
    a `checksum` field (hex-encoded SHA-256 of the file's raw bytes).
    Whole file in one request -- fine for ordinary photos/short clips;
    see the chunked views below for large/slow uploads.

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

            body, status = _stage_validated_file(tmp_path, upload.name)
            return Response(body, status=status)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def _chunk_scratch_dir(upload_id):
    return os.path.join(settings.UPLOAD_CHUNK_SCRATCH_DIR, str(upload_id))


class InitChunkedUploadView(APIView):
    """POST /api/upload/chunked/init/ -- start a new chunked upload.

    Body: {filename, total_size, checksum} (checksum: hex SHA-256 of the
    WHOLE file, re-verified once every chunk has arrived). Returns
    {upload_id, chunk_size, total_chunks} -- the frontend must split the
    file into `total_chunks` pieces of at most `chunk_size` bytes each
    (the last one naturally smaller) and PUT each to UploadChunkView.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        filename = (request.data.get('filename') or '').strip()
        checksum = (request.data.get('checksum') or '').strip().lower()
        total_size = request.data.get('total_size')

        if not filename or _classify_extension(filename) is None:
            return Response({
                'error': f'Unsupported or missing filename "{filename}" -- '
                         f'expected an image, video, or .zip file.',
            }, status=400)
        if not _CHECKSUM_RE.match(checksum):
            return Response(
                {'error': 'A hex-encoded SHA-256 "checksum" field is required.'},
                status=400,
            )
        try:
            total_size = int(total_size)
        except (TypeError, ValueError):
            return Response({'error': 'A numeric "total_size" (bytes) field is required.'}, status=400)
        if total_size <= 0:
            return Response({'error': '"total_size" must be positive.'}, status=400)
        if total_size > settings.UPLOAD_MAX_FILE_SIZE_BYTES:
            return Response({
                'error': f'File too large ({total_size} bytes, max '
                         f'{settings.UPLOAD_MAX_FILE_SIZE_BYTES}).',
            }, status=413)

        chunk_size = settings.UPLOAD_CHUNK_SIZE_BYTES
        total_chunks = math.ceil(total_size / chunk_size)
        session = UploadSession.objects.create(
            user=request.user, filename=filename, total_size=total_size,
            checksum=checksum, chunk_size=chunk_size, total_chunks=total_chunks,
        )
        return Response({
            'upload_id': str(session.upload_id),
            'chunk_size': chunk_size,
            'total_chunks': total_chunks,
        }, status=201)


class UploadChunkView(APIView):
    """PUT /api/upload/chunked/<upload_id>/chunk/<index>/ -- multipart
    form with a `chunk` field (raw bytes for this piece) and a
    `checksum` field (hex SHA-256 of just this chunk). Idempotent:
    re-sending the same index overwrites it, so a client can safely
    retry a single failed chunk without restarting the whole upload."""

    permission_classes = (IsAuthenticated,)
    parser_classes = (MultiPartParser,)

    def put(self, request, upload_id, index):
        session = get_object_or_404(UploadSession, upload_id=upload_id, user=request.user)
        if session.status != UploadSession.STATUS_IN_PROGRESS:
            return Response({'error': 'This upload session is no longer in progress.'}, status=409)
        if not (0 <= index < session.total_chunks):
            return Response(
                {'error': f'Chunk index out of range (0-{session.total_chunks - 1}).'},
                status=400,
            )

        chunk_file = request.FILES.get('chunk')
        if chunk_file is None:
            return Response({'error': 'No chunk provided (expected multipart field "chunk").'}, status=400)
        if chunk_file.size > session.chunk_size:
            return Response(
                {'error': f'Chunk too large ({chunk_file.size} bytes, max {session.chunk_size}).'},
                status=413,
            )

        checksum = (request.data.get('checksum') or '').strip().lower()
        if not _CHECKSUM_RE.match(checksum):
            return Response(
                {'error': 'A hex-encoded SHA-256 "checksum" field is required for this chunk.'},
                status=400,
            )

        chunk_dir = _chunk_scratch_dir(session.upload_id)
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_path = os.path.join(chunk_dir, f'{index}.part')

        hasher = hashlib.sha256()
        with open(chunk_path, 'wb') as out:
            for piece in chunk_file.chunks():
                hasher.update(piece)
                out.write(piece)

        if hasher.hexdigest() != checksum:
            os.remove(chunk_path)
            return Response(
                {'error': 'Chunk checksum mismatch -- please resend this chunk.'},
                status=400,
            )

        UploadChunk.objects.update_or_create(
            session=session, chunk_index=index, defaults={'size': chunk_file.size},
        )
        received = sorted(session.chunks.values_list('chunk_index', flat=True))
        return Response({'received_chunks': received, 'total_chunks': session.total_chunks}, status=200)


class ChunkedUploadStatusView(APIView):
    """GET /api/upload/chunked/<upload_id>/status/ -- which chunks have
    already been received, so a frontend can resume after a dropped
    connection or a page reload without re-sending chunks it already
    successfully delivered."""

    permission_classes = (IsAuthenticated,)

    def get(self, request, upload_id):
        session = get_object_or_404(UploadSession, upload_id=upload_id, user=request.user)
        received = sorted(session.chunks.values_list('chunk_index', flat=True))
        return Response({
            'status': session.status,
            'received_chunks': received,
            'total_chunks': session.total_chunks,
            'chunk_size': session.chunk_size,
        })


class CompleteChunkedUploadView(APIView):
    """POST /api/upload/chunked/<upload_id>/complete/ -- reassembles every
    received chunk in order, re-verifies the whole-file checksum
    declared at init time (defense in depth beyond the per-chunk checks
    -- also catches e.g. chunks reassembled out of order), then runs the
    exact same content-verification/staging path as the single-shot
    endpoint. Chunk scratch data is removed afterward either way."""

    permission_classes = (IsAuthenticated,)

    def post(self, request, upload_id):
        session = get_object_or_404(UploadSession, upload_id=upload_id, user=request.user)
        if session.status != UploadSession.STATUS_IN_PROGRESS:
            return Response({'error': 'This upload session is no longer in progress.'}, status=409)

        received = sorted(session.chunks.values_list('chunk_index', flat=True))
        expected = list(range(session.total_chunks))
        if received != expected:
            missing = sorted(set(expected) - set(received))
            return Response(
                {'error': 'Not all chunks have been received.', 'missing_chunks': missing},
                status=400,
            )

        chunk_dir = _chunk_scratch_dir(session.upload_id)
        hasher = hashlib.sha256()
        bytes_written = 0
        fd, tmp_path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'wb') as out:
                for i in range(session.total_chunks):
                    chunk_path = os.path.join(chunk_dir, f'{i}.part')
                    with open(chunk_path, 'rb') as cf:
                        while True:
                            block = cf.read(1024 * 1024)
                            if not block:
                                break
                            hasher.update(block)
                            out.write(block)
                            bytes_written += len(block)

            if bytes_written != session.total_size:
                session.status = UploadSession.STATUS_FAILED
                session.save(update_fields=['status'])
                return Response({
                    'error': f'Reassembled size ({bytes_written} bytes) does not match '
                             f'the declared total_size ({session.total_size}).',
                }, status=400)

            if hasher.hexdigest() != session.checksum:
                session.status = UploadSession.STATUS_FAILED
                session.save(update_fields=['status'])
                return Response({'error': 'Reassembled file checksum mismatch.'}, status=400)

            body, status_code = _stage_validated_file(tmp_path, session.filename)
            session.status = (
                UploadSession.STATUS_COMPLETED if status_code < 400 else UploadSession.STATUS_FAILED
            )
            session.save(update_fields=['status'])
            return Response(body, status=status_code)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            shutil.rmtree(chunk_dir, ignore_errors=True)
