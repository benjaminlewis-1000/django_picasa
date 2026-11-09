#! /usr/bin/env python

# Trains the planned ML orientation classifier (ImageFile.
# RotationSignalSource.ML_CLASSIFIER -- see CLAUDE.md's "rotation-review
# ML classifier" open TODO). Design, discussed with the user before any
# code (2026-11-09): a self-supervised "predict the synthetic rotation"
# pretext task -- the standard approach when there's no real labeled
# orientation dataset (there isn't one here: the human rotation-review
# queue has 870 FLAGGED/0 CONFIRMED rows at last count, far too little
# and too skewed toward hard cases to train on directly).
#
# Ground truth: a real ImageFile, decoded through the exact same
# EXIF-orientation (+ any human-confirmed rotation override) path the
# rest of this app already uses (common.open_img_oriented), is assumed
# correctly oriented -- true for the overwhelming majority of real
# photos, the same assumption the classic "predicting image rotations"
# self-supervision literature makes, and the same risk-tolerance this
# project's own earlier orientation-signal investigation already
# accepted. Only the small, already-flagged-as-possibly-wrong
# RotationScanState.FLAGGED population is excluded from the sample, to
# avoid training on a known-suspect label.
#
# Architecture: MobileNetV3-Small, ImageNet-pretrained, 4-way head (see
# rotation_classifier_model.build_model()'s own docstring for why this
# backbone specifically -- cheap enough to match this project's existing
# all-ONNX-CPU-inference convention at eventual production scale, and
# pretrained rather than trained from scratch since there's no large
# labeled dataset to justify the latter).
#
# Training regime: step-based (not epoch-based) and resumable, because
# this runs on shared CPU alongside this project's own backfill/
# remediation jobs -- intended to be launched in short bursts whenever
# there's real headroom (--max-minutes), not as one long uninterrupted
# run. The train/val image-id split is decided once (first launch) and
# persisted into the checkpoint, specifically so a later resumed
# invocation can't accidentally re-sample a different split and leak
# validation images into training (or vice versa) across runs.
# torch.set_num_threads is capped (--threads, default 8 of this host's
# 28 cores) so a training burst doesn't by itself starve whatever else
# is running -- the whole point of "launch when there's headroom" is
# defeated if the job then consumes all of it itself.
#
# Deliberately NOT done here: wiring this classifier into
# rotation_audit.py as a real, live RotationSignalSource.ML_CLASSIFIER
# signal. That's a separate, later integration step (needs its own
# staleness/versioning field distinct from the landmark signal's source-
# hash, per the already-recorded Open TODOs note) -- this command only
# trains and exports the model.

import json
import os
import time

import numpy as np
from django.core.management.base import BaseCommand

from common.open_img_oriented import open_img_oriented
from face_manager.rotation_classifier_model import (
    INPUT_SIZE, NUM_CLASSES, apply_synthetic_rotation, build_model,
    letterbox_to_square, to_model_input,
)
from filepopulator.models import ImageFile

DEFAULT_CHECKPOINT_PATH = '/models/rotation_classifier/checkpoint.pt'
DEFAULT_ONNX_PATH = '/models/rotation_classifier/rotation_classifier.onnx'
DEFAULT_SPLIT_PATH = '/models/rotation_classifier/split.json'

CHECKPOINT_EVERY_STEPS = 200
EVAL_EVERY_STEPS = 1000
LR_DECAY_EVERY_STEPS = 2000
LR_DECAY_FACTOR = 0.5


def _sample_image_ids(num_train, num_val, seed):
    """Random, reproducible (given a fixed seed) split of real,
    not-already-flagged-as-possibly-rotated ImageFile ids. Plain Python
    random rather than the rest of this codebase's usual
    .order_by('?') -- that shuffles at the DB level per QUERY, which
    would give a different split on every call; this needs exactly one
    frozen split, persisted and reused across every resumed training
    invocation (see this module's own docstring)."""
    import random
    eligible_ids = list(
        ImageFile.objects.exclude(
            rotation_scan_state=ImageFile.RotationScanState.FLAGGED,
        ).values_list('id', flat=True)
    )
    rng = random.Random(seed)
    rng.shuffle(eligible_ids)
    total_needed = num_train + num_val
    if len(eligible_ids) < total_needed:
        raise ValueError(
            f"Only {len(eligible_ids)} eligible images exist, need {total_needed} "
            f"(--num-train-images + --num-val-images) -- lower one of those."
        )
    return eligible_ids[:num_train], eligible_ids[num_train:total_needed]


def _load_or_create_split(split_path, num_train, num_val, seed):
    if os.path.exists(split_path):
        with open(split_path) as f:
            data = json.load(f)
        return data['train_ids'], data['val_ids']
    train_ids, val_ids = _sample_image_ids(num_train, num_val, seed)
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, 'w') as f:
        json.dump({'train_ids': train_ids, 'val_ids': val_ids}, f)
    return train_ids, val_ids


def _decode_upright(filename, extra_rotation_degrees):
    """Real, already-EXIF-(and-any-confirmed-rotation-)corrected pixel
    content for one image file -- the ground-truth 'upright' example
    this whole training scheme is built on. Takes plain values (not an
    ImageFile instance -- see _RotationDataset's own docstring for why)
    so this never touches the ORM/a live DB connection. Returns None on
    a real decode failure (a handful of genuinely corrupted/missing
    files are expected at this library's scale -- skipped, not fatal,
    same tolerance this project's other bulk-scan commands already
    have)."""
    try:
        arr = open_img_oriented(filename, as_numpy=True, extra_rotation_degrees=extra_rotation_degrees)
    except (FileNotFoundError, OSError):
        return None
    return arr


def _augment_brightness_contrast(arr_float, rng):
    # Deliberately NOT a flip (see rotation_classifier_model's own
    # letterbox/pad docstring area and this command's design notes --
    # a horizontal or vertical flip can interact ambiguously with a
    # 4-way rotation label on near-symmetric content) and deliberately
    # NOT a crop that could remove the edge/corner cues (a horizon line,
    # a ceiling) this task actually depends on. Mild brightness/contrast
    # jitter only -- enough to stop the model keying on exact pixel
    # statistics without touching anything geometry-related.
    brightness = rng.uniform(0.85, 1.15)
    contrast = rng.uniform(0.85, 1.15)
    mean = arr_float.mean()
    arr_float = (arr_float - mean) * contrast + mean
    arr_float = arr_float * brightness
    return np.clip(arr_float, 0.0, 255.0)


class _RotationDataset:
    """Not a torch.utils.data.Dataset subclass -- this project's other
    CPU-bound ONNX pipelines (video_face_pipeline.py etc.) don't use
    torch DataLoader either, and a plain iterable keeps this testable
    without a torch import at collection time. __getitem__ still works
    (sequence protocol), so torch.utils.data.DataLoader(this) works
    unmodified regardless.

    Deliberately resolves every image's (filename, confirmed-rotation-
    override) up front in __init__ -- a single bulk query, run once in
    the MAIN process -- rather than querying the ORM from __getitem__.
    A DataLoader with num_workers > 0 forks worker PROCESSES that
    inherit whatever DB connection object already existed at fork time;
    letting multiple forked workers independently run ORM queries over
    that same inherited connection is exactly the class of bug already
    documented elsewhere in this codebase (CLAUDE.md's "--parallel test
    worker" investigation) -- concurrent use of one psycopg2 connection
    across processes is unsafe, not just slow. Resolving everything to
    plain Python values up front means __getitem__ never touches the
    ORM/a DB connection at all, so this is safe regardless of worker
    count."""

    def __init__(self, image_ids, train, seed=0):
        self.train = train
        self._rng = __import__('random').Random(seed)
        rows = ImageFile.objects.filter(id__in=image_ids).values_list(
            'id', 'filename', 'rotation_scan_state', 'proposed_rotation_degrees',
        )
        self._file_info = {}
        for image_id, filename, scan_state, proposed_degrees in rows:
            extra = proposed_degrees if (
                scan_state == ImageFile.RotationScanState.CONFIRMED and proposed_degrees is not None
            ) else None
            self._file_info[image_id] = (filename, extra)
        self.image_ids = [iid for iid in image_ids if iid in self._file_info]
        # Train: one random rotation per visit (law-of-large-numbers
        # balanced across many steps/epochs, standard for this kind of
        # synthetic-rotation self-supervision -- no need to pre-
        # materialize all 4 variants per image). Val: every image fixed
        # to all 4 rotations, for a complete, deterministic, reproducible
        # metric -- built once here rather than on the fly.
        if not train:
            self._val_pairs = [(iid, k) for iid in self.image_ids for k in range(NUM_CLASSES)]

    def __len__(self):
        return len(self.image_ids) if self.train else len(self._val_pairs)

    def __getitem__(self, idx):
        import torch

        if self.train:
            image_id = self.image_ids[idx]
            k = self._rng.randrange(NUM_CLASSES)
        else:
            image_id, k = self._val_pairs[idx]

        filename, extra_rotation_degrees = self._file_info[image_id]
        arr = _decode_upright(filename, extra_rotation_degrees)
        if arr is None:
            # Skip a real decode failure by substituting a different
            # valid sample rather than crashing the whole DataLoader
            # worker over a handful of bad files at this library's
            # scale -- same tolerance this project's other bulk scans
            # already have for corrupted/missing originals.
            return self.__getitem__((idx + 1) % len(self))

        rotated = apply_synthetic_rotation(arr, k)
        square = letterbox_to_square(rotated, size=INPUT_SIZE)
        if self.train:
            square = _augment_brightness_contrast(square.astype(np.float32), self._rng).astype(np.uint8)
        tensor = to_model_input(square)
        return torch.from_numpy(tensor), k


class Command(BaseCommand):
    help = (
        "Trains the planned ML orientation classifier via self-supervised "
        "synthetic-rotation prediction. Step-based and resumable -- intended "
        "to be run in short bursts (--max-minutes) whenever there's real CPU "
        "headroom, not as one long run."
    )

    def add_arguments(self, parser):
        parser.add_argument('--max-minutes', type=float, default=60.0,
                             help='Stop (cleanly, after checkpointing) once this much wall '
                                  'time has elapsed in THIS invocation. Launch again later to '
                                  'resume from the saved checkpoint.')
        parser.add_argument('--num-train-images', type=int, default=24000)
        parser.add_argument('--num-val-images', type=int, default=2000)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--num-workers', type=int, default=2,
                             help='DataLoader worker processes. Kept low by default -- these '
                                  'compete for the same CPU headroom this whole command is '
                                  'supposed to be respecting.')
        parser.add_argument('--threads', type=int, default=8,
                             help="torch's own intra-op thread count, capped well below this "
                                  "host's full core count so a training burst doesn't by "
                                  "itself consume all the headroom it was launched to use.")
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--checkpoint-path', type=str, default=DEFAULT_CHECKPOINT_PATH)
        parser.add_argument('--onnx-export-path', type=str, default=DEFAULT_ONNX_PATH)
        parser.add_argument('--split-path', type=str, default=DEFAULT_SPLIT_PATH)
        parser.add_argument('--split-seed', type=int, default=20261109)
        parser.add_argument('--fresh', action='store_true',
                             help='Ignore any existing checkpoint and start a brand new model '
                                  '(the train/val split is still reused if --split-path already '
                                  'exists, to keep results comparable).')

    def handle(self, *args, **options):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        torch.set_num_threads(options['threads'])

        checkpoint_path = options['checkpoint_path']
        onnx_path = options['onnx_export_path']
        deadline = time.time() + options['max_minutes'] * 60

        train_ids, val_ids = _load_or_create_split(
            options['split_path'], options['num_train_images'], options['num_val_images'],
            options['split_seed'],
        )
        self.stdout.write(f"Split: {len(train_ids)} train images, {len(val_ids)} val images "
                           f"({len(val_ids) * NUM_CLASSES} val examples across all 4 rotations).")

        train_ds = _RotationDataset(train_ids, train=True)
        val_ds = _RotationDataset(val_ids, train=False)
        train_loader = DataLoader(
            train_ds, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'],
        )

        model = build_model(pretrained=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_DECAY_EVERY_STEPS, gamma=LR_DECAY_FACTOR,
        )
        global_step = 0
        best_val_acc = 0.0

        if not options['fresh'] and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            global_step = ckpt['global_step']
            best_val_acc = ckpt.get('best_val_acc', 0.0)
            self.stdout.write(f"Resumed from step {global_step} (best val acc so far: "
                               f"{best_val_acc:.3f})")
        else:
            self.stdout.write("Starting a fresh model.")

        criterion = nn.CrossEntropyLoss()

        def save_checkpoint():
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'global_step': global_step,
                'best_val_acc': best_val_acc,
            }, checkpoint_path)

        def export_onnx():
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            model.eval()
            dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
            # dynamo=False: torch 2.13's new default (dynamo=True) exporter
            # needs the `onnxscript` package, not installed in this image
            # (confirmed via a real smoke-test crash, 2026-11-09) -- the
            # legacy TorchScript-based exporter doesn't need it and is
            # otherwise equivalent for a model this simple.
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=['image'], output_names=['logits'],
                dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
                opset_version=17, dynamo=False,
            )
            model.train()

        @torch.no_grad()
        def run_eval():
            model.eval()
            correct = 0
            total = 0
            per_class_correct = [0] * NUM_CLASSES
            per_class_total = [0] * NUM_CLASSES
            for x, y in val_loader:
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += int((preds == y).sum())
                total += int(y.shape[0])
                for c in range(NUM_CLASSES):
                    mask = y == c
                    per_class_total[c] += int(mask.sum())
                    per_class_correct[c] += int((preds[mask] == c).sum())
            model.train()
            acc = correct / total if total else 0.0
            per_class_acc = [
                (per_class_correct[c] / per_class_total[c] if per_class_total[c] else None)
                for c in range(NUM_CLASSES)
            ]
            return acc, per_class_acc

        model.train()
        ran_out_of_time = False
        for x, y in train_loader:
            if time.time() >= deadline:
                ran_out_of_time = True
                break

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 20 == 0:
                self.stdout.write(f"  step {global_step}: loss={loss.item():.4f} "
                                   f"lr={scheduler.get_last_lr()[0]:.2e}")

            if global_step % CHECKPOINT_EVERY_STEPS == 0:
                save_checkpoint()

            if global_step % EVAL_EVERY_STEPS == 0:
                acc, per_class_acc = run_eval()
                self.stdout.write(f"  [eval @ step {global_step}] val_acc={acc:.4f} "
                                   f"per_class={per_class_acc}")
                if acc > best_val_acc:
                    best_val_acc = acc
                    save_checkpoint()
                    export_onnx()
                    self.stdout.write(f"  new best val_acc={acc:.4f} -- exported {onnx_path}")

        # Real bug found via a smoke test (2026-11-09): this final save/eval
        # used to save the checkpoint BEFORE running this final eval, so a
        # genuinely new best accuracy reached only here (the loop's own
        # periodic eval, every EVAL_EVERY_STEPS, might not land exactly on
        # the last step) was computed but never actually written into the
        # checkpoint's best_val_acc -- a resumed run would then report
        # "best val acc so far: 0.000" even after a real improving run.
        # best_val_acc must be updated BEFORE the save that persists it.
        acc, per_class_acc = run_eval()
        is_new_best = acc > best_val_acc
        if is_new_best:
            best_val_acc = acc
        save_checkpoint()
        self.stdout.write(
            f"\nStopped at step {global_step} ({'time budget reached' if ran_out_of_time else 'train loader exhausted -- one full epoch'}).\n"
            f"Final val_acc={acc:.4f} per_class={per_class_acc}. Best val_acc so far: {best_val_acc:.4f}.\n"
            f"Checkpoint saved to {checkpoint_path}. Run this command again to resume."
        )
        if is_new_best:
            export_onnx()
            self.stdout.write(f"Exported final ONNX model to {onnx_path}")
