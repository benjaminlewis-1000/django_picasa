"""Model definition + preprocessing for the planned ML orientation
classifier (ImageFile.RotationSignalSource.ML_CLASSIFIER -- see the
Open TODOs entry in CLAUDE.md: "the rotation-review ML classifier...
planned as a second signal source alongside the current landmark-
geometry one", not yet built). The landmark-geometry signal
(rotation_geometry.py) is free and already validated, but only applies
to images with at least one detected face (62,354 of 270,470
NOT_SCANNED images at last count have zero faces at all -- see
rotation_audit.py's own docstring) -- this is the second signal meant
to cover that gap, and faceless content generally.

Kept deliberately separate from rotation_audit.py/rotation_geometry.py
(both pure-math, zero ML dependency, and the module whose SOURCE HASH
gates the landmark signal's own re-scan eligibility -- see
rotation_audit._rotation_scan_logic_hash()'s own docstring and the
CLAUDE.md note that a model-based classifier must NOT share that same
hash, since re-running the landmark scan on every unrelated edit here
costs nothing, but re-running real inference on the whole backlog on
every edit to THIS file would be genuinely expensive). This module and
its own staleness/versioning (once the classifier is actually wired
into a live signal) are tracked independently.

Everything here is pure (no Django/DB dependency) so it's directly
importable from the training command, a future live-inference path,
and tests without any of them dragging in the others.
"""
import numpy as np
from PIL import Image, ImageOps

NUM_CLASSES = 4  # degrees of CLOCKWISE rotation already baked into a
                  # training example, relative to the real upright
                  # orientation: class k means "rotate CCW by 90*k
                  # degrees to undo it and recover upright." Matches
                  # np.rot90(img, k)'s own convention (k applied CCW),
                  # so build_training_example() and undo_rotation()
                  # agree by construction, not by a separately-stated
                  # convention that could drift out of sync.

INPUT_SIZE = 224  # matches the ImageNet-pretrained backbone's expected
                   # input -- see build_model()'s own reasoning.

# ImageNet normalization stats -- required to get useful signal out of
# ImageNet-pretrained backbone weights; arbitrary/mismatched
# normalization would waste most of the pretraining's value.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Letterbox pad color -- mid-gray (same value for all 3 channels before
# normalization) rather than black: a black border is itself a strong,
# consistent edge/gradient feature right at the image boundary that a
# CNN can latch onto as a shortcut (e.g. "which side has more black"
# instead of genuinely reading the photo's content for orientation
# cues) -- a real, known failure mode for letterboxed training data.
# Mid-gray is comparatively low-signal in either direction.
PAD_COLOR = (128, 128, 128)


def letterbox_to_square(image_rgb_uint8, size=INPUT_SIZE):
    """Resize-and-pad an HxWx3 uint8 RGB array to size x size, preserving
    aspect ratio (never stretching) -- the resulting aspect ratio of the
    PRE-pad content is itself a real orientation signal (a landscape-
    shaped photo that's actually rotated 90 degrees becomes portrait-
    shaped on disk, which is exactly one of the cues this classifier
    should be able to use), which a plain stretch-to-square resize would
    throw away. Returns an size x size x 3 uint8 array."""
    pil_img = Image.fromarray(image_rgb_uint8)
    padded = ImageOps.pad(pil_img, (size, size), color=PAD_COLOR, centering=(0.5, 0.5))
    return np.asarray(padded, dtype=np.uint8)


def to_model_input(image_rgb_uint8):
    """size x size x 3 uint8 RGB -> 3 x size x size float32, ImageNet-
    normalized -- the exact tensor shape/scaling build_model()'s
    backbone expects. Pure numpy (no torch dependency) so this is
    callable from a plain-numpy test or a future ONNX-based inference
    path without importing torch at all."""
    arr = image_rgb_uint8.astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))  # HWC -> CHW


def apply_synthetic_rotation(image_rgb_uint8, k):
    """np.rot90 with k in {0,1,2,3} -- the self-supervised pretext-task
    label generator. k=0 is upright (the real ground truth, since the
    base image is assumed already-correctly-oriented -- see the
    training command's own sampling docstring for why that assumption
    is safe); k=1/2/3 are synthetic 90/180/270-degree CCW rotations,
    each a real, officially-wrong example the model must learn to
    detect and correctly label."""
    return np.ascontiguousarray(np.rot90(image_rgb_uint8, k=k))


def undo_rotation_degrees(predicted_class):
    """The real-world correction angle (CCW degrees) a predicted class
    implies -- e.g. if the model predicts class 1 (meaning the image
    was rotated 90 degrees CW from upright, consistent with
    apply_synthetic_rotation's np.rot90(k=1) convention), undoing it
    means rotating 90 degrees CCW, i.e. +90 under rotation_geometry's
    own atan2/CCW-positive convention. Not yet consumed anywhere live
    (this classifier isn't wired into rotation_audit.py as a real
    signal source yet -- training/validation only, see CLAUDE.md) --
    kept here now so the one real convention this model's output means
    is defined in exactly one place, ready for that integration."""
    return {0: 0.0, 1: 90.0, 2: 180.0, 3: -90.0}[predicted_class]


def build_model(pretrained=True):
    """MobileNetV3-Small (torchvision), ImageNet-pretrained backbone,
    classifier head replaced for NUM_CLASSES outputs.

    Chosen over a custom from-scratch CNN for two concrete reasons,
    not just "it's standard": (1) ImageNet pretraining already encodes
    real, transferable low/mid-level visual structure (edges, textures,
    object parts) that a from-scratch CPU-trained model would need a
    much larger dataset and many more epochs to approximate on its own
    -- this project doesn't have (and doesn't need) a human-labeled
    orientation dataset, see the training command's self-supervised
    synthetic-rotation approach, so leaning on existing pretraining is
    the efficient use of the real compute budget available (CPU-only,
    shared with this project's own backfill/remediation jobs). (2) it's
    specifically a *mobile/efficiency* architecture (~2.5M params,
    depthwise-separable convs) -- this project already standardizes on
    cheap ONNX+onnxruntime CPU inference for every other vision model
    in the pipeline (insightface's det_500m/det_10g, LVFace), and this
    one needs to run over the same hundreds-of-thousands-of-images
    scale eventually -- a heavier backbone (ResNet50, etc.) would cost
    meaningfully more at that inference scale for a 4-way classification
    task that doesn't need deep feature richness."""
    import torch.nn as nn
    import torchvision

    if pretrained:
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = torchvision.models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model
