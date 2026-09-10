#! /usr/bin/env python
"""Phase 3 video face extraction: detect faces across a video's sampled
frames, track them frame-to-frame, cluster the tracks under TWO
independently-trained embedding models (insightface + facenet-pytorch),
and union-merge the two clusterings into final per-video identity groups
-- one Face row per group, not per track or per frame.

This design (and every constant/threshold below) comes directly out of a
real investigation against 13 real videos, documented in CLAUDE.md's
Phase 3 section -- see that write-up for the full reasoning, including
why union-merge (not intersection/"agreement-gating") was chosen, why
deinterlacing/2x sampling/det_10g re-detection matter, and the known
false-merge risk this design deliberately accepts (the user's own call:
"I'm not aiming for complete accuracy, just an idea of who's likely in
videos").

The pure data-shape functions (_iou_track, _cluster_track_faces,
_union_merge_groups, _trimmed_centroid) take plain lists/arrays, not
Django objects, and have no video-decode or ONNX dependency -- they're
unit-tested directly with synthetic embeddings. VideoFaceExtractor is the
Django/ONNX-touching orchestration layer around them, exercised only by
the real-fixture (slow-tagged) integration test.
"""
from django.conf import settings
from django.core.files.base import ContentFile
from face_manager.models import Person, Face
from filepopulator.models import VideoFile
from io import BytesIO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import av
import cv2
import insightface.app.common
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from insightface.utils import face_align
import json
import numpy as np
import os
import subprocess
import torch
import torchvision.ops.boxes as bops

# ---------------------------------------------------------------------------
# Constants -- every value here was empirically validated this session
# against 13 real videos (see CLAUDE.md), not guessed.
# ---------------------------------------------------------------------------
IOU_THRESH = 0.30
BOX_GROW_PCT = 0.25
MAX_GAP_SAMPLES = 1
STRIDE_DIVISOR = 2  # "2x density" -- halves the originally-scoped stride
N_BEST_FRAMES_PER_TRACK = 2
CLUSTER_COS_THRESHOLD = 0.5
CLUSTER_LINKAGE = 'average'
CENTROID_OUTLIER_SIM_FLOOR = 0.35
DISALLOWED_DIST = 1e6
REDETECT_PAD_PCT = 0.6
FACENET_INPUT_SIZE = 160


def cos_to_euclidean(cos_thresh):
    return float(np.sqrt(max(0.0, 2 - 2 * cos_thresh)))


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Pure functions -- plain data in, plain data out, no Django/ONNX/video I/O.
# ---------------------------------------------------------------------------

def grow_box(bbox, pct):
    l, t, r, b = bbox
    w, h = r - l, b - t
    return [l - w * pct, t - h * pct, r + w * pct, b + h * pct]


def _iou_track(sampled_detections, iou_thresh=IOU_THRESH, box_grow_pct=BOX_GROW_PCT,
                max_gap=MAX_GAP_SAMPLES):
    """sampled_detections: list of (sample_idx, boxes, scores, kps_list)
    tuples, one per sampled frame, in increasing sample_idx order.
    Returns a list of track dicts: {'boxes', 'scores', 'frames', 'kps'},
    each a same-indexed list across only the samples that track matched.
    Frame-to-frame linking uses the Hungarian algorithm (optimal
    one-shot assignment per frame, not greedy nearest-match) on IOU of
    box-grown boxes, exactly matching the reconciliation logic already
    used for image re-detection (face_extract_encode.py)."""
    open_tracks, closed_tracks = [], []
    for sample_idx, boxes, scores, kps_list in sampled_detections:
        grown_boxes = [grow_box(b, box_grow_pct) for b in boxes]
        matched_det, matched_tracks = set(), set()
        if open_tracks and grown_boxes:
            existing_grown = torch.tensor([t['last_box_grown'] for t in open_tracks], dtype=torch.float32)
            detect_t = torch.tensor(grown_boxes, dtype=torch.float32)
            iou_mat = bops.distance_box_iou(existing_grown, detect_t).numpy()
            rows, cols = linear_sum_assignment(-iou_mat)
            for r, c in zip(rows, cols):
                if iou_mat[r, c] < iou_thresh:
                    continue
                open_tracks[r]['boxes'].append(boxes[c])
                open_tracks[r]['scores'].append(scores[c])
                open_tracks[r]['frames'].append(sample_idx)
                open_tracks[r]['kps'].append(kps_list[c])
                open_tracks[r]['last_box_grown'] = grown_boxes[c]
                open_tracks[r]['misses'] = 0
                matched_tracks.add(r)
                matched_det.add(c)
        for r in range(len(open_tracks) - 1, -1, -1):
            if r not in matched_tracks:
                open_tracks[r]['misses'] += 1
                if open_tracks[r]['misses'] > max_gap:
                    closed_tracks.append(open_tracks.pop(r))
        for c, box in enumerate(boxes):
            if c not in matched_det:
                open_tracks.append({
                    'boxes': [box], 'scores': [scores[c]], 'frames': [sample_idx],
                    'kps': [kps_list[c]], 'last_box_grown': grown_boxes[c], 'misses': 0,
                })
    closed_tracks.extend(open_tracks)
    for t in closed_tracks:
        t['span'] = (t['frames'][0], t['frames'][-1])
    closed_tracks.sort(key=lambda t: t['span'][0])
    return closed_tracks


def _cluster_track_faces(face_embeddings, face_track_id, track_spans,
                          cos_threshold=CLUSTER_COS_THRESHOLD, linkage=CLUSTER_LINKAGE):
    """face_embeddings: (n_faces, D) array, one row per representative
    frame (multiple per track). face_track_id: (n_faces,) int array
    mapping each row to its track index. track_spans: dict/list of
    track_idx -> (first_sample, last_sample). Returns dict track_idx ->
    group_label (int, not globally meaningful, just local to this
    clustering pass).

    Must-link (same-track pairs forced to distance 0) + cannot-link
    (temporally-overlapping different-track pairs forced to a disallowed
    distance) -- see CLAUDE.md for why both are necessary (complete/
    average linkage's own strict worst-pair-must-clear-threshold rule
    otherwise splits a single track's own natural pose variation apart
    without the must-link fix)."""
    face_embeddings = np.asarray(face_embeddings)
    face_track_id = np.asarray(face_track_id)
    n_faces = len(face_embeddings)
    if n_faces == 0:
        return {}
    normed = face_embeddings / np.linalg.norm(face_embeddings, axis=1, keepdims=True)
    dist = squareform(pdist(normed.astype(np.float32), metric='euclidean'))
    for i in range(n_faces):
        for j in range(n_faces):
            if i == j:
                continue
            ti, tj = face_track_id[i], face_track_id[j]
            if ti == tj:
                dist[i, j] = 0.0
                continue
            span_i, span_j = track_spans[ti], track_spans[tj]
            overlaps = not (span_i[1] < span_j[0] or span_j[1] < span_i[0])
            if overlaps:
                dist[i, j] = DISALLOWED_DIST

    if n_faces == 1:
        labels = np.array([0])
    else:
        distance_threshold = cos_to_euclidean(cos_threshold)
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold,
            linkage=linkage, metric='precomputed',
        ).fit_predict(dist)

    track_to_label = {}
    for i, label in enumerate(labels):
        ti = int(face_track_id[i])
        track_to_label.setdefault(ti, set()).add(int(label))
    # Same-track faces are must-linked (forced distance 0), so in
    # practice every face in one track always lands in the same label;
    # take the first defensively rather than assert, since a future
    # change to the must-link logic should degrade gracefully here.
    return {ti: next(iter(labels_set)) for ti, labels_set in track_to_label.items()}


def _union_merge_groups(n_tracks, *groupings):
    """groupings: any number of dict track_idx -> label (from separate
    clustering passes, e.g. one per embedding model). Returns a dict
    track_idx -> final_group_id, where two tracks land in the same final
    group if EITHER input grouping placed them together (union, not
    intersection -- the user's own explicit call: fewer final groups,
    some wrong merges acceptable, rather than agreement-gating)."""
    parent = list(range(n_tracks))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for grouping in groupings:
        by_label = {}
        for ti, label in grouping.items():
            by_label.setdefault(label, []).append(ti)
        for members in by_label.values():
            for m in members[1:]:
                union(members[0], m)

    return {ti: find(ti) for ti in range(n_tracks)}


def _trimmed_centroid(embeddings, sim_floor=CENTROID_OUTLIER_SIM_FLOOR):
    """embeddings: (n, D) array pooled from every representative frame of
    every track in one final union-merge group. Computes the plain mean,
    drops any row whose cosine similarity to that mean is below
    sim_floor, and recomputes the mean from survivors (one-pass trimmed
    mean, per the user's own call: "trim parts of the group that fall
    below the threshold"). Falls back to the untrimmed mean if trimming
    would remove everything (shouldn't happen in practice -- a group's
    own mean is never farther than sim_floor from every one of its own
    members unless the group is pathologically diverse)."""
    embeddings = np.asarray(embeddings)
    mean = embeddings.mean(axis=0)
    sims = np.array([cos_sim(e, mean) for e in embeddings])
    survivors = embeddings[sims >= sim_floor]
    if len(survivors) == 0:
        return mean
    return survivors.mean(axis=0)


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------

def ffprobe_info(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,field_order',
         '-show_entries', 'stream_side_data=rotation', '-of', 'json', path],
        capture_output=True, text=True,
    )
    data = json.loads(out.stdout)
    stream = data['streams'][0]
    width, height = stream['width'], stream['height']
    # avg_frame_rate (real nb_frames/duration) rather than r_frame_rate
    # (the container's declared NOMINAL rate) -- confirmed via a real
    # 2026-09-09 survey these can differ substantially and in ways that
    # silently corrupt every frame_idx-based timestamp downstream:
    # interlaced PAL content reports r_frame_rate as the FIELD rate (50)
    # while avg_frame_rate correctly reports the real FRAME rate (25) --
    # since this pipeline already deinterlaces (one output frame per
    # input frame), using r_frame_rate there computed every timestamp at
    # exactly half the true elapsed time. Some phone videos also declare
    # a high nominal rate (e.g. 120, a slow-mo capability) while actually
    # delivering ~30fps content. Falls back to r_frame_rate only if
    # avg_frame_rate is missing/zero (not observed in a 482-video survey,
    # but a real container could omit it).
    num, den = stream['r_frame_rate'].split('/')
    r_fps = float(num) / float(den) if float(den) else 0
    avg_num, avg_den = stream.get('avg_frame_rate', '0/1').split('/')
    avg_fps = float(avg_num) / float(avg_den) if float(avg_den) else 0
    fps = avg_fps if avg_fps > 0 else r_fps
    field_order = stream.get('field_order', 'unknown')
    rotation = 0
    for sd in stream.get('side_data_list', []):
        if 'rotation' in sd:
            rotation = int(sd['rotation'])
    if rotation in (90, -90, 270, -270):
        width, height = height, width
    return width, height, fps, field_order, rotation


def ffmpeg_frame_iterator(path, width, height, vf_filter=None, seek_seconds=0):
    """seek_seconds, if given, places -ss BEFORE -i -- a fast, approximate
    seek (snaps to the nearest preceding keyframe, not frame-accurate),
    used by callers that already know roughly where they need to start
    reading and don't need the very first yielded frame to be an exact
    timestamp (e.g. backfill_video_thumbnail_timestamps.py, which
    pixel-matches across a window of candidates rather than trusting any
    single frame's assumed index)."""
    frame_size = width * height * 3
    cmd = ['ffmpeg', '-v', 'error']
    if seek_seconds > 0:
        cmd += ['-ss', str(seek_seconds)]
    cmd += ['-i', path]
    if vf_filter:
        cmd += ['-vf', vf_filter]
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_size * 2)
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def _mean_abs_pixel_diff(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def extract_frame_near_timestamp(path, target_seconds, avg_fps, field_order, rotation,
                                  box, reference_bgr, window_seconds=0.5):
    """On-demand exact-frame retrieval for the video-face "full context"
    viewer (api/views.py's _extract_video_face_frame) -- NOT used by the
    main extraction pipeline itself, which already reads frames
    sequentially in original decode order and has no need for this.

    Real benchmarking (2026-09-09, see CLAUDE.md) found ffmpeg's own -ss
    seeking (either placed before or after -i) has a real ~30% mismatch
    rate against the frame a face's stored thumbnail/box actually came
    from -- even when fed the exact correct target timestamp, ffmpeg's
    internal frame-selection after a seek doesn't always agree with true
    sequential decode order for some files in this library.

    Fixed by using PyAV directly, and by not trusting any single index/
    pts computation to land on the exactly-right frame at all: seek to
    the nearest keyframe before a small window around the target, decode
    every frame in that window, and return whichever one's own box-crop
    is the closest pixel match to the face's ALREADY-STORED thumbnail.
    This absorbed every real failure mode found during benchmarking --
    off-by-a-few-frame rounding between how a timestamp was originally
    derived (sequential frame count / avg_fps at pipeline time) and how
    it's re-derived here, PTS start-time offsets on some older containers
    (confirmed on real Pre_camcorder-era MPGs, ~3 frames on one real
    file), and a stored target that slightly overshoots the real last
    frame (falls back to the closest frame that actually exists) --
    without needing to get the seek/index math exactly right, since the
    stored thumbnail is the real ground truth either way. Only decodes
    roughly 2*window_seconds*avg_fps frames, not the whole video.

    Returns a raw BGR numpy array (full frame, not cropped), or None if
    no frame in the window decodes at all.
    """
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'

        graph = None
        if field_order not in ('progressive', 'unknown'):
            graph = av.filter.Graph()
            buf = graph.add_buffer(template=stream)
            yadif = graph.add('yadif', 'mode=0')
            buf.link_to(yadif)
            sink = graph.add('buffersink')
            yadif.link_to(sink)
            graph.configure()

        # Frame time must be computed relative to the stream's OWN start
        # pts, not raw pts=0 -- some real files (old camcorder-era MPGs)
        # have a nonzero start offset that otherwise silently shifts
        # every frame's computed time by a few frames' worth.
        start_pts = stream.start_time if stream.start_time is not None else 0
        seek_target = max(0.0, target_seconds - window_seconds)
        container.seek(int(seek_target * av.time_base), backward=True, any_frame=False, stream=None)

        best_arr = None
        best_diff = None
        for packet in container.demux(stream):
            for frame in packet.decode():
                candidates = [frame]
                if graph is not None:
                    graph.push(frame)
                    candidates = []
                    while True:
                        try:
                            candidates.append(graph.pull())
                        except Exception:
                            break
                for out_frame in candidates:
                    pts = out_frame.pts if out_frame.pts is not None else 0
                    frame_time = float((pts - start_pts) * stream.time_base)
                    if frame_time > target_seconds + window_seconds:
                        return best_arr
                    if frame_time < target_seconds - window_seconds:
                        continue
                    arr = out_frame.to_ndarray(format='bgr24')
                    # ffmpeg's raw pipe (used elsewhere in this file)
                    # auto-applies the container's rotation side-data;
                    # PyAV does not, so it's applied explicitly here.
                    if rotation:
                        arr = np.rot90(arr, k=round(rotation / 90) % 4)
                    crop = VideoFaceExtractor._square_thumbnail(arr, box)
                    d = _mean_abs_pixel_diff(crop, reference_bgr)
                    if d is not None and (best_diff is None or d < best_diff):
                        best_diff = d
                        best_arr = arr
        return best_arr
    finally:
        container.close()


def sample_stride(fps):
    return max(1, round(fps * (20 / 30.0) / STRIDE_DIVISOR))


# ---------------------------------------------------------------------------
# Django/ONNX orchestration layer
# ---------------------------------------------------------------------------

class VideoFaceExtractor(object):
    """One instance per batch run (reuses loaded models + the gallery
    cache across many videos, same convention as FaceExtractor/
    faceAssigner elsewhere in this codebase)."""

    def __init__(self):
        self.det_model = self._load_det500m_with_fallback()
        self.det_model.prepare(ctx_id=-1, det_size=(640, 640))
        rec_app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
        rec_app.prepare(ctx_id=-1, det_size=(640, 640))
        self.rec_model = rec_app.models['recognition']
        self.det10g_model = rec_app.models['detection']

        from facenet_pytorch import InceptionResnetV1
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

        self.blank_face_person = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
        self._face_assigner = None  # lazily built on first use, reused across videos

    DET500M_PATH = '/root/.insightface/models/buffalo_s/det_500m.onnx'

    @classmethod
    def _load_det500m_with_fallback(cls):
        """buffalo_l (used for recognition/det_10g above) and antelopev2
        are already pre-populated in this project's model volume; buffalo_s
        (only used here, for the lightweight bulk-sampling detector) was
        not, and model_zoo.get_model() -- unlike FaceAnalysis(name=...) --
        only checks the file exists rather than downloading it, so this
        crashed the first time this pipeline ran for real (2026-09-08).
        Self-heals by triggering the same auto-download FaceAnalysis(name=
        'buffalo_s') already does internally, once, if the direct load
        fails -- so a fresh deploy/wiped model volume doesn't hit this
        same silent trap again. Requires outbound internet access for
        that one-time ~120MB download (confirmed available in this
        environment); if that's ever not the case, this still fails, just
        with a clearer error than the original bare AssertionError."""
        if not os.path.exists(cls.DET500M_PATH):
            settings.LOGGER.warning(
                "buffalo_s model pack missing at %s -- downloading it now "
                "(one-time, ~120MB).", cls.DET500M_PATH
            )
            FaceAnalysis(name='buffalo_s').prepare(ctx_id=-1)
        return model_zoo.get_model(cls.DET500M_PATH)

    @property
    def face_assigner(self):
        if self._face_assigner is None:
            from assign_faces import faceAssigner
            self._face_assigner = faceAssigner()
            self._face_assigner.load_encodings()
        return self._face_assigner

    def _encode_insightface(self, frame, kps):
        kps_arr = np.array(kps, dtype=np.float32).reshape(5, 2)
        face = insightface.app.common.Face(kps=kps_arr)
        return self.rec_model.get(frame, face)

    def _encode_facenet(self, frame, kps):
        kps_arr = np.array(kps, dtype=np.float32).reshape(5, 2)
        aligned = face_align.norm_crop(frame, kps_arr, image_size=112)
        aligned = cv2.resize(aligned, (FACENET_INPUT_SIZE, FACENET_INPUT_SIZE))
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - 127.5) / 128.0
        with torch.no_grad():
            emb = self.facenet_model(tensor)
        return emb.squeeze(0).numpy()

    def _redetect_in_crop(self, frame, box, pad_pct=REDETECT_PAD_PCT):
        l, t, r, b = box
        w, h = r - l, b - t
        l2 = max(0, int(l - w * pad_pct))
        t2 = max(0, int(t - h * pad_pct))
        r2 = min(frame.shape[1], int(r + w * pad_pct))
        b2 = min(frame.shape[0], int(b + h * pad_pct))
        crop = frame[t2:b2, l2:r2]
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            return None
        bboxes, kpss = self.det10g_model.detect(crop, max_num=0, metric='default')
        if bboxes.shape[0] == 0:
            return None
        best = int(np.argmax(bboxes[:, 4]))
        box_out = [bboxes[best, 0] + l2, bboxes[best, 1] + t2,
                   bboxes[best, 2] + l2, bboxes[best, 3] + t2]
        kps_out = (kpss[best] + np.array([l2, t2])).tolist()
        return box_out, kps_out

    @staticmethod
    def _square_thumbnail(frame, box, extension_mult=2):
        """Adapted from FaceExtractor.get_square_face_img (same margin
        logic/thumbnail size), without the ImageFile-specific type check
        -- frame here is a raw decoded video frame, not tied to any
        Django model."""
        img_h, img_w, _ = frame.shape
        bb_l, bb_t, bb_r, bb_b = box
        bb_l = max(0, bb_l)
        bb_t = max(0, bb_t)
        bb_r = min(bb_r, img_w)
        bb_b = min(bb_b, img_h)

        face_h = bb_b - bb_t
        face_w = bb_r - bb_l
        face_center_vert = (bb_b - bb_t) // 2 + bb_t
        face_center_horiz = (bb_r - bb_l) // 2 + bb_l

        vert_margin = min(face_center_vert, img_h - face_center_vert)
        horiz_margin = min(face_center_horiz, img_w - face_center_horiz)
        detection_max_dim = max(face_h, face_w)
        max_allowable_margin = min(vert_margin, horiz_margin)
        ideal_thumbnail_margin = detection_max_dim * extension_mult // 2
        actual_margin = min(ideal_thumbnail_margin, max_allowable_margin)
        actual_margin = max(actual_margin, detection_max_dim // 2)

        chip_l = int(face_center_horiz - actual_margin)
        chip_r = int(face_center_horiz + actual_margin)
        chip_t = int(face_center_vert - actual_margin)
        chip_b = int(face_center_vert + actual_margin)

        left_pad = right_pad = top_pad = bot_pad = 0
        if chip_l < 0:
            left_pad, chip_l = -chip_l, 0
        if chip_r > img_w:
            right_pad, chip_r = chip_r - img_w, img_w
        if chip_t < 0:
            top_pad, chip_t = -chip_t, 0
        if chip_b > img_h:
            bot_pad, chip_b = chip_b - img_h, img_h

        thumb = frame[chip_t:chip_b, chip_l:chip_r]
        thumb = np.pad(thumb, ((top_pad, bot_pad), (left_pad, right_pad), (0, 0)), 'constant')
        # NOTE: no BGR->RGB conversion here, unlike FaceExtractor's own
        # get_square_face_img() (which this was adapted from) -- that
        # function's input is already RGB (PIL-decoded, via
        # common.open_img_oriented), so its own cvtColor(BGR2RGB) call is
        # actually undoing that RGB order back to BGR, which cv2.imencode
        # below then correctly re-interprets as BGR and writes a normal-
        # looking JPEG -- a double-swap that happens to cancel out. This
        # function's `frame` is genuinely BGR to begin with (raw ffmpeg
        # pipe, -pix_fmt bgr24), so it's already in the order cv2.imencode
        # expects -- adding the same cvtColor call here (as an earlier
        # version of this function did) applies only ONE swap, producing
        # a thumbnail with red/blue channels genuinely reversed.
        return cv2.resize(thumb, settings.FACE_THUMBNAIL_SIZE)

    def _detect_and_track(self, video_path, width, height, fps, vf_filter):
        stride = sample_stride(fps)
        sampled = []
        idx = 0
        for frame in ffmpeg_frame_iterator(video_path, width, height, vf_filter=vf_filter):
            if idx % stride == 0:
                bboxes, kpss = self.det_model.detect(frame, max_num=0, metric='default')
                boxes = bboxes[:, :4].tolist() if bboxes.shape[0] else []
                scores = bboxes[:, 4].tolist() if bboxes.shape[0] else []
                kps_list = kpss.tolist() if kpss is not None and bboxes.shape[0] else []
                sampled.append((idx, boxes, scores, kps_list))
            idx += 1
        # idx is now the true total decoded frame count -- returned so the
        # caller can derive a REAL fps (total_frames / known duration)
        # instead of trusting ffprobe's r_frame_rate/avg_frame_rate, either
        # of which can be wrong for a given file in different ways (see
        # process_video's real_fps comment for the real case that proved
        # this necessary).
        return _iou_track(sampled), stride, idx

    def _pool_representative_frames(self, video_path, width, height, vf_filter, tracks):
        """Picks each track's best N_BEST_FRAMES_PER_TRACK frames (by
        det_score*area), re-decodes just those frames, re-detects each
        with det_10g, and encodes with both models. Mutates each track
        dict in place, adding 'reps': [{'emb_if','emb_fn','frame_idx','box'}...]."""
        needed_frames = set()
        for t in tracks:
            quality = []
            for box, score in zip(t['boxes'], t['scores']):
                l, tt, r, b = box
                area = max(0, r - l) * max(0, b - tt)
                quality.append(score * area)
            best_idxs = np.argsort(quality)[::-1][:N_BEST_FRAMES_PER_TRACK]
            t['best_idxs'] = best_idxs
            for i in best_idxs:
                needed_frames.add(t['frames'][i])

        frame_pixels = {}
        idx = 0
        for frame in ffmpeg_frame_iterator(video_path, width, height, vf_filter=vf_filter):
            if idx in needed_frames:
                frame_pixels[idx] = frame.copy()
            idx += 1

        for t in tracks:
            reps = []
            for i in t['best_idxs']:
                frame_idx = t['frames'][i]
                frame = frame_pixels[frame_idx]
                orig_box = t['boxes'][i]
                redet = self._redetect_in_crop(frame, orig_box)
                box_use, kps_use = redet if redet is not None else (orig_box, t['kps'][i])
                emb_if = self._encode_insightface(frame, kps_use)
                emb_fn = self._encode_facenet(frame, kps_use)
                reps.append({'emb_if': emb_if, 'emb_fn': emb_fn, 'frame_idx': frame_idx,
                             'box': box_use, 'kps': kps_use})
            t['reps'] = reps
        return frame_pixels

    def _classify_and_pick_thumbnail(self, face, pooled_reps, frame_pixels, fps):
        """face is an already-saved Face row (source_video_file/box/kps/
        thumbnail all set from a provisional -- largest-box -- frame,
        face_encoding_512 set to the group centroid). Runs the real,
        untouched classify_unassigned() against it; if that produces a
        confident match (poss_ident1 not one of the ignore sentinels),
        re-picks the thumbnail as whichever pooled frame has the highest
        sim_99th against that SPECIFIC person's own gallery, and updates
        the Face row's box/kps/thumbnail to that frame. Returns nothing;
        mutates and re-saves `face` in place if the thumbnail changes."""
        self.face_assigner.classify_unassigned(face)
        face.refresh_from_db()

        if face.poss_ident1 is None or face.poss_ident1.person_name in settings.IGNORED_NAMES:
            return  # no confident match -- keep the provisional (largest-box) thumbnail

        fa = self.face_assigner
        person_id = face.poss_ident1_id
        if person_id not in fa.gallery_offsets:
            return
        lo, hi = fa.gallery_offsets[person_id]

        best_rep, best_sim = None, -2.0
        for rep in pooled_reps:
            emb = rep['emb_if']
            query_norm = np.linalg.norm(emb)
            similarity = (emb @ fa.all_embeddings[:, lo:hi]) / (fa.all_norms[lo:hi] * query_norm)
            sim_99th = float(np.percentile(similarity, 99))
            if sim_99th > best_sim:
                best_sim, best_rep = sim_99th, rep

        if best_rep is None:
            return

        frame = frame_pixels[best_rep['frame_idx']]
        self._set_face_box_and_thumbnail(
            face, frame, best_rep['box'], best_rep['kps'], best_rep['frame_idx'] / fps
        )
        face.save()

    @staticmethod
    def _clamp_to_duration(video_obj, seconds, label):
        """A computed video_*_timestamp_seconds value should never exceed
        the video's own real duration -- but a real bug (2026-09-09,
        confirmed on /videos/Our_Home_Videos/2020/20200618_221605.mp4,
        source_video_file id 368) produced timestamps up to ~4x the real
        199.5s duration, root cause not yet fully diagnosed (raw decoded
        frame count vs. avg_frame_rate*duration mismatch under
        investigation). This is a safety net, not a fix for whatever
        produces an out-of-range value in the first place -- clamps to
        the video's own duration and logs loudly, so a face still gets a
        usable (if approximate) timestamp instead of one that points past
        the end of the file, while leaving a clear trail to investigate
        rather than silently masking a real miscalculation."""
        duration = video_obj.duration_seconds
        if duration is None or seconds <= duration:
            return seconds
        settings.LOGGER.warning(
            "video_face_pipeline: %s timestamp %.2fs exceeds video %s's own "
            "duration %.2fs (file=%s) -- clamping to duration. This points "
            "at a real timestamp-computation bug, not expected behavior.",
            label, seconds, video_obj.id, duration, video_obj.filename,
        )
        return duration

    def _set_face_box_and_thumbnail(self, face, frame, box, kps, timestamp_seconds=None):
        l, t, r, b = box
        img_h, img_w, _ = frame.shape
        face.box_left = max(1, int(l))
        face.box_top = max(1, int(t))
        face.box_right = min(img_w, max(face.box_left + 1, int(r)))
        face.box_bottom = min(img_h, max(face.box_top + 1, int(b)))
        face.kps = np.asarray(kps, dtype=float).reshape(-1).tolist()
        if timestamp_seconds is not None:
            face.video_thumbnail_frame_seconds = self._clamp_to_duration(
                face.source_video_file, timestamp_seconds, 'video_thumbnail_frame_seconds'
            )

        thumbnail = self._square_thumbnail(frame, box)
        is_success, buffer_img = cv2.imencode('.jpg', thumbnail)
        temp_thumb = BytesIO(buffer_img)
        temp_thumb.seek(0)
        thumb_filename = f'video{face.source_video_file_id}_group_{face.id or "new"}.jpg'
        face.face_thumbnail.save(thumb_filename, ContentFile(temp_thumb.read()), save=False)
        temp_thumb.close()

    def process_video(self, video_obj: VideoFile):
        """Runs the full pipeline for one VideoFile and creates one Face
        row per final union-merge group. Returns the list of created
        Face objects."""
        video_path = video_obj.filename
        width, height, fps, field_order, _rotation = ffprobe_info(video_path)
        deinterlace = field_order not in ('progressive', 'unknown')
        vf_filter = 'yadif=0' if deinterlace else None

        tracks, stride, total_frames = self._detect_and_track(video_path, width, height, fps, vf_filter)
        if not tracks:
            return []

        # real_fps (actual decoded frame count / the video's own known
        # duration) replaces ffprobe's fps for every frame-index-to-
        # seconds conversion below -- confirmed necessary 2026-09-09 on a
        # real file (source_video_file id 368) where ffprobe's own
        # avg_frame_rate metadata (nb_frames/duration) understated the
        # true decoded frame count by exactly 4x (5981 claimed vs. 23935
        # actually decoded, for a real 199.5s video) -- a genuine
        # container/metadata quirk (likely a slow-motion capture whose
        # declared duration doesn't match its real frame count), not a
        # "wrong ffprobe field selected" bug like the earlier interlaced-
        # PAL case this file's r_frame_rate/avg_frame_rate split was
        # originally built for. Neither ffprobe field is reliably correct
        # for every file, but the actual decode already happened above
        # (for detection) at zero extra cost, so deriving fps from ITS
        # real frame count is unconditionally more trustworthy than
        # either metadata field. Falls back to ffprobe's fps only if
        # duration_seconds isn't known (shouldn't happen for anything
        # that reached this point, but avoids a divide-by-zero/None).
        duration = video_obj.duration_seconds
        real_fps = (total_frames / duration) if duration else fps
        if duration and abs(real_fps - fps) / fps > 0.05:
            settings.LOGGER.warning(
                "video_face_pipeline: video %s (%s) real fps %.2f (from "
                "%d decoded frames / %.2fs duration) differs from "
                "ffprobe's %.2f by more than 5%% -- using the real value.",
                video_obj.id, video_obj.filename, real_fps, total_frames,
                duration, fps,
            )
        fps = real_fps

        frame_pixels = self._pool_representative_frames(video_path, width, height, vf_filter, tracks)

        # Flatten to one row per representative frame for clustering.
        embs_if, embs_fn, face_track_id = [], [], []
        for ti, t in enumerate(tracks):
            for rep in t['reps']:
                embs_if.append(rep['emb_if'])
                embs_fn.append(rep['emb_fn'])
                face_track_id.append(ti)
        track_spans = {ti: t['span'] for ti, t in enumerate(tracks)}

        groups_if = _cluster_track_faces(embs_if, face_track_id, track_spans)
        groups_fn = _cluster_track_faces(embs_fn, face_track_id, track_spans)
        final_group_of = _union_merge_groups(len(tracks), groups_if, groups_fn)

        groups = {}
        for ti, gid in final_group_of.items():
            groups.setdefault(gid, []).append(ti)

        created = []
        for members in groups.values():
            pooled_reps = [rep for m in members for rep in tracks[m]['reps']]
            if not pooled_reps:
                continue
            pooled_embs_if = np.array([r['emb_if'] for r in pooled_reps])
            centroid = _trimmed_centroid(pooled_embs_if)

            # Provisional thumbnail: largest box among the pool.
            def _box_area(rep):
                l, t, r, b = rep['box']
                return max(0, r - l) * max(0, b - t)
            provisional = max(pooled_reps, key=_box_area)
            provisional_frame = frame_pixels[provisional['frame_idx']]

            first_sample = min(tracks[m]['span'][0] for m in members)
            last_sample = max(tracks[m]['span'][1] for m in members)

            face = Face()
            face.source_video_file = video_obj
            face.declared_name = self.blank_face_person
            face.dateTakenUTC = video_obj.dateTakenUTC
            face.reencoded = True
            face.written_to_photo_metadata = False
            face.face_encoding_512 = centroid.tolist()
            face.video_first_timestamp_seconds = self._clamp_to_duration(
                video_obj, first_sample / fps, 'video_first_timestamp_seconds'
            )
            face.video_last_timestamp_seconds = self._clamp_to_duration(
                video_obj, last_sample / fps, 'video_last_timestamp_seconds'
            )
            self._set_face_box_and_thumbnail(
                face, provisional_frame, provisional['box'], provisional['kps'],
                provisional['frame_idx'] / fps
            )
            face.save()

            self._classify_and_pick_thumbnail(face, pooled_reps, frame_pixels, fps)
            created.append(face)

        return created
