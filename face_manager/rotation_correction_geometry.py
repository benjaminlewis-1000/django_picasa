"""Pure geometry for APPLYING a confirmed rotation correction -- the
counterpart to face_manager.rotation_geometry (which only ever DETECTS
a likely rotation, never corrects one). No Django/DB dependency, so
it's directly unit-testable.

Corrections are snapped to exact 90-degree increments (90/-90/180) and
applied via PIL's Image.transpose(ROTATE_90/ROTATE_180/ROTATE_270) --
the same lossless, resampling-free mechanism common/open_img_oriented.py
already uses for EXIF orientation -- rather than a general-angle
Image.rotate(expand=True), since the real validated use case (see
CLAUDE.md's 2026-11-03/11-05/11-06 write-up) is specifically camera-
sensor-orientation misses (a clean quarter/half turn), not an arbitrary
tilt. A future ML classifier could in principle propose a non-90-
multiple angle (ImageFile.proposed_rotation_degrees isn't restricted
to one), but it still gets snapped to the nearest canonical bucket
before any correction is actually applied.

snap_to_canonical_rotation()/CANONICAL_ROTATIONS_DEG themselves live in
common/open_img_oriented.py, not here -- that's the single shared
image-decode chokepoint every consumer (face extraction, serving,
ingestion) already goes through for EXIF orientation, and the
confirmed rotation is applied the same way, right after it (see
ImageFile.confirmed_rotation_override_degrees). Re-exported here so
existing importers of this module don't need to know that.

The exact point-transform formulas below were derived empirically
(not assumed) against real PIL.Image.transpose() output -- see the
2026-11-05 session transcript for the verification script -- and are
locked in by PointTransformTests below.
"""
from common.open_img_oriented import CANONICAL_ROTATIONS_DEG, snap_to_canonical_rotation

__all__ = [
    'CANONICAL_ROTATIONS_DEG', 'snap_to_canonical_rotation',
    'transformed_image_size', 'transform_point', 'transform_box', 'transform_kps',
]


def transformed_image_size(old_width, old_height, snapped_degrees):
    """(new_width, new_height) after applying snapped_degrees. 90/-90
    swap width and height (a quarter turn); 180 does not.
    """
    if snapped_degrees == 180.0:
        return old_width, old_height
    return old_height, old_width


def transform_point(x, y, old_width, old_height, snapped_degrees):
    """Maps a pixel coordinate (x, y) in the OLD (old_width x
    old_height) image to its corresponding coordinate in the image
    that results from applying snapped_degrees via PIL's transpose().
    snapped_degrees must be one of CANONICAL_ROTATIONS_DEG (call
    snap_to_canonical_rotation() first if it might not be).
    """
    if snapped_degrees == 90.0:
        # PIL Image.transpose(Image.ROTATE_90)
        return y, (old_width - 1) - x
    if snapped_degrees == -90.0:
        # PIL Image.transpose(Image.ROTATE_270)
        return (old_height - 1) - y, x
    if snapped_degrees == 180.0:
        return (old_width - 1) - x, (old_height - 1) - y
    raise ValueError(f"snapped_degrees must be one of {CANONICAL_ROTATIONS_DEG}, got {snapped_degrees}")


def transform_box(left, top, right, bottom, old_width, old_height, snapped_degrees):
    """Maps an axis-aligned box to its new axis-aligned box. Exact (no
    re-fitting/slack needed) since every supported correction is an
    exact 90-degree multiple -- an axis-aligned box always maps to
    another axis-aligned box under one of these transforms.
    """
    x1, y1 = transform_point(left, top, old_width, old_height, snapped_degrees)
    x2, y2 = transform_point(right, bottom, old_width, old_height, snapped_degrees)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def transform_kps(kps_flat, old_width, old_height, snapped_degrees):
    """Maps a flat 10-float kps array (5 points) the same way. Returns
    a flat list, matching Face.kps's own stored shape.
    """
    pts = [(kps_flat[i], kps_flat[i + 1]) for i in range(0, len(kps_flat), 2)]
    transformed = [transform_point(x, y, old_width, old_height, snapped_degrees) for x, y in pts]
    return [coord for point in transformed for coord in point]
