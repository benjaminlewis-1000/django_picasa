#!/usr/bin/env python
"""Generates synthetic (non-personal-photo) test fixtures for CI.

Run once, locally, whenever fixtures need regenerating -- the output is
committed as static files under ci_fixtures/, so CI itself never needs
this script or its extra deps (piexif, pillow-heif) at all.

Produces, relative to this file's directory:
  test_imgs_filepopulate/   -- synthetic stand-in for the real fixture
                                directory filepopulator/face_manager/api
                                tests point at via
                                settings.FILEPOPULATOR_VAL_DIRECTORY
  corrupted/                -- synthetic truncated/broken JPEGs
  heic_stub/                -- one synthetic .heic file

None of these contain real photo content -- every image is procedurally
generated (solid colors / simple gradients), not sourced from a personal
photo library.
"""
import io
import os

import numpy as np
import piexif
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def _solid_jpeg_bytes(width=200, height=150, color=(120, 60, 200), exif_bytes=None):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    # A diagonal stripe breaks the symmetry, useful if orientation is
    # ever asserted on later.
    for i in range(min(width, height)):
        img[i, i] = (255, 255, 255)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    if exif_bytes:
        pil_img.save(buf, format="JPEG", exif=exif_bytes)
    else:
        pil_img.save(buf, format="JPEG")
    return buf.getvalue()


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def build_test_imgs_filepopulate():
    root = os.path.join(HERE, "test_imgs_filepopulate")

    # Loose top-level files the existing tests reference by name.
    write(os.path.join(root, "has_face_tags.jpg"), _solid_jpeg_bytes(color=(200, 120, 60)))
    write(os.path.join(root, "has_same_faces.jpg"), _solid_jpeg_bytes(color=(200, 120, 60)))
    write(os.path.join(root, "gps.jpg"), _solid_jpeg_bytes(color=(60, 200, 120)))
    write(os.path.join(root, "question_date_taken.jpg"), _solid_jpeg_bytes(color=(60, 120, 200)))
    write(os.path.join(root, "tmpmv.jpg"), _solid_jpeg_bytes(color=(90, 90, 90)))
    write(os.path.join(root, "2019-11-03 16.44.50.jpg"), _solid_jpeg_bytes(color=(10, 200, 10)))

    # naming/good -- valid jpeg-ish extensions in various cases, spaces,
    # unicode, and a subdirectory with repeated basenames across dirs.
    good = os.path.join(root, "naming", "good")
    for name, color in [
        ("1.JPG", (10, 10, 200)),
        ("2.jpg", (20, 20, 190)),
        ("3.JPEG", (30, 30, 180)),
        ("4.jpeg", (40, 40, 170)),
        ("5.jpg", (50, 50, 160)),
        ("6.JpEg", (60, 60, 150)),
        ("space in filename.jpg", (70, 70, 140)),
        ("unicode_ãàÓ.jpg", (80, 80, 130)),
    ]:
        write(os.path.join(good, name), _solid_jpeg_bytes(color=color))

    challenge_dir = os.path.join(good, "challenge dírectôry_with_repeats")
    for name, color in [
        ("1.JPG", (11, 22, 33)),
        ("4.jpeg", (44, 55, 66)),
        ("9.jpeg", (99, 88, 77)),
        ("10.jpg", (10, 20, 30)),
    ]:
        write(os.path.join(challenge_dir, name), _solid_jpeg_bytes(color=color))

    # naming/bad -- files that exist but should be rejected: wrong/missing
    # extensions, non-image content, extension in the middle of the name.
    bad = os.path.join(root, "naming", "bad")
    write(os.path.join(bad, "1.png"), _solid_jpeg_bytes())  # jpeg bytes, wrong extension
    write(os.path.join(bad, "1.jpeeg"), _solid_jpeg_bytes())  # misspelled extension
    write(os.path.join(bad, "2.jp"), _solid_jpeg_bytes())  # truncated extension
    write(os.path.join(bad, "3.txt"), b"not an image at all")
    write(os.path.join(bad, "4.jpg.txt"), _solid_jpeg_bytes())
    write(os.path.join(bad, "5.jpg830"), _solid_jpeg_bytes())

    # orientation/ -- one file per EXIF Orientation tag value (1-8).
    # Not asserted on by any currently-active test, but existing tests
    # walk this directory, so it needs to exist with real files in it.
    orient_dir = os.path.join(root, "orientation")
    for orientation in range(1, 9):
        exif_dict = {"0th": {piexif.ImageIFD.Orientation: orientation}}
        exif_bytes = piexif.dump(exif_dict)
        write(
            os.path.join(orient_dir, f"2_{orientation}.jpg"),
            _solid_jpeg_bytes(color=(orientation * 20, 100, 200 - orientation * 20), exif_bytes=exif_bytes),
        )


def build_corrupted():
    # Both fixtures below raise OSError on decode (confirmed against a
    # real libjpeg-turbo decode, not just by construction) -- that's all
    # the tests that use these actually assert on. Real corrupted photos
    # in production raised two differently-worded OSErrors ("truncated"
    # vs "broken data stream") depending on exactly where the damage
    # landed; reliably reproducing the *second* wording synthetically
    # turned out to need more fragile byte-surgery than it's worth here,
    # since no test distinguishes between the two messages.
    root = os.path.join(HERE, "corrupted")
    good = _solid_jpeg_bytes(width=300, height=300, color=(180, 40, 40))
    write(os.path.join(root, "truncated_a.jpg"), good[: len(good) - 200])

    good2 = _solid_jpeg_bytes(width=300, height=300, color=(40, 40, 180))
    write(os.path.join(root, "truncated_b.jpg"), good2[: len(good2) - 50])


def build_heic():
    import pillow_heif

    root = os.path.join(HERE, "heic_stub")
    img = Image.fromarray(
        (np.ones((150, 200, 3), dtype=np.uint8) * np.array([90, 160, 210], dtype=np.uint8))
    )
    heif_file = pillow_heif.from_pillow(img)
    os.makedirs(root, exist_ok=True)
    heif_file.save(os.path.join(root, "synthetic.heic"), quality=80)


if __name__ == "__main__":
    build_test_imgs_filepopulate()
    build_corrupted()
    build_heic()
    print(f"Fixtures written under {HERE}")
