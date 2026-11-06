"""
Pure geometry helpers for the (validated-but-not-yet-built) face-landmark
orientation-detection signal -- see CLAUDE.md's 2026-11-03 "a cheap (no new
ML model) orientation-detection signal" write-up for the background and the
validation numbers this is built on.

Everything here is pure (no Django/DB/model dependency) and operates on
plain floats/lists so it's directly unit-testable and reusable from both a
future audit command and the eventual correction pipeline.

Angle convention: degrees, computed via atan2(dy, dx) on the eye-to-eye
vector (left eye -> right eye, kps indices 0 and 1). 0 degrees means the
eyes are horizontal (upright face). The sign/magnitude also indicates which
way a face appears rotated, consistent with atan2's standard range of
(-180, 180].
"""
import math
from statistics import median

import numpy as np


def eye_angle_degrees(kps):
    """Eye-to-eye orientation angle for a single face's 5-point kps.

    kps is a flat sequence of 10 floats (x0, y0, x1, y1, ..., x4, y4) or an
    equivalent 5x2 sequence -- left eye is point 0, right eye is point 1.
    """
    pts = np.asarray(kps, dtype=float).reshape(-1, 2)
    left_eye, right_eye = pts[0], pts[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return math.degrees(math.atan2(dy, dx))


def circular_distance(angle_a, angle_b):
    """Smallest angular distance between two angles, in [0, 180].

    Correctly handles the 0/360 wraparound -- e.g. circular_distance(2, 358)
    is 4, not 356.
    """
    diff = abs(angle_a - angle_b) % 360
    return min(diff, 360 - diff)


def wrap_to_pm180(angle):
    """Wrap an arbitrary angle into (-180, 180]."""
    wrapped = (angle + 180) % 360 - 180
    # (x % 360) is always in [0, 360), so the only edge case is exactly -180
    # wrapping to -180 instead of the canonical +180 -- normalize that one.
    if wrapped == -180:
        wrapped = 180
    return wrapped


def circular_mean(angles):
    """Mean of a list of angles (degrees), correct across the wraparound.

    Implemented via unit-vector averaging (atan2 of the mean sin/cos),
    not a plain arithmetic mean, so e.g. mean([355, 5]) == 0, not 180.
    """
    radians = np.radians(angles)
    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))
    return math.degrees(math.atan2(mean_sin, mean_cos))


def circular_median(angles):
    """Median of a list of angles (degrees), correct across the wraparound.

    Only meaningful/stable when the angles form a tight-ish cluster (span
    well under 180 degrees) -- which is exactly the case this is used for
    (faces on the same, consistently-oriented photo). Implemented by
    unwrapping every angle relative to the circular mean (so they land on a
    single continuous number line with no wrap discontinuity nearby), taking
    an ordinary median, then wrapping the result back.
    """
    if len(angles) == 1:
        return angles[0]
    center = circular_mean(angles)
    unwrapped = [center + wrap_to_pm180(a - center) for a in angles]
    return wrap_to_pm180(median(unwrapped))


# Faces must agree within this many degrees (circularly) to be trusted as a
# single orientation signal without falling back to a scan/classifier.
DEFAULT_AGREEMENT_THRESHOLD_DEG = 20.0


def aggregate_orientation(angles, agreement_threshold_deg=DEFAULT_AGREEMENT_THRESHOLD_DEG):
    """Combine several faces' eye-angle estimates into one orientation call.

    Returns (estimate_degrees, inlier_count) on success, or None if there's
    not enough agreement to trust a verdict (caller should fall back to a
    scan/classifier in that case).

    Policy (per the agreed design):
      - 0 or 1 angles: not enough signal, return None.
      - 2 angles: both must agree within agreement_threshold_deg (circular
        distance) or we bail; if they agree, return their circular mean.
      - 3+ angles: take a first-pass circular median as a robust center,
        drop any angle whose circular distance to that center exceeds
        agreement_threshold_deg (outliers), then return the circular median
        of just the remaining inliers. If fewer than 2 inliers survive,
        bail (not enough agreement).
    """
    angles = list(angles)
    n = len(angles)
    if n < 2:
        return None

    if n == 2:
        if circular_distance(angles[0], angles[1]) <= agreement_threshold_deg:
            return (circular_mean(angles), 2)
        return None

    first_pass_center = circular_median(angles)
    inliers = [a for a in angles if circular_distance(a, first_pass_center) <= agreement_threshold_deg]
    if len(inliers) < 2:
        return None
    return (circular_median(inliers), len(inliers))


# Below this many degrees of circular distance from 0, treat an estimate
# as "basically upright" -- normal head-tilt noise, not a rotated photo.
# The real validation sample (see CLAUDE.md's 2026-11-03 write-up) found
# 99% of real faces within ~26 degrees of horizontal and nothing anywhere
# near 90 degrees -- 15 sits comfortably inside that noise floor while
# still well short of a genuine quarter-turn.
UPRIGHT_TOLERANCE_DEG = 15.0

# Ordering used to decide whether a NEW proposal is allowed to overwrite
# an EXISTING one for the same image: a write only takes effect if its
# source's priority is >= the current proposal's source priority. Human
# confirm/dismiss is handled separately (always terminal, never
# overwritten by a re-scan) -- this ordering only governs automated
# signal sources competing with each other. Higher index = higher
# priority (a future ML classifier should always win over a pure
# landmark-geometry guess; multi-face agreement is more trustworthy than
# a single lonely face).
ROTATION_SOURCE_PRIORITY = {
    'landmark_single_face': 0,
    'landmark_multi_face': 1,
    'ml_classifier': 2,
}


def should_overwrite_rotation_proposal(existing_source, new_source):
    """Whether a new automated proposal from new_source is allowed to
    replace an existing one from existing_source (per ROTATION_SOURCE_
    PRIORITY). existing_source of None (never scanned, or a prior human
    dismissal with no proposal) always allows the write. Never call this
    to decide whether to overwrite a HUMAN-sourced proposal -- that's
    handled by checking rotation_scan_state, not this priority table.
    """
    if existing_source is None:
        return True
    return ROTATION_SOURCE_PRIORITY[new_source] >= ROTATION_SOURCE_PRIORITY[existing_source]
