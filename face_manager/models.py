from django.db import models
from django.db.models import Q
from django.db.models.expressions import RawSQL
from django.conf import settings
from django.forms import ModelForm
from django.core.validators import MinValueValidator, MaxValueValidator, RegexValidator
from django.core.exceptions import ValidationError
from django.core.validators import *
from datetime import datetime
from django.utils import timezone
from django.contrib.postgres.fields import ArrayField
from pgvector.django import HalfVectorField
import cv2
import numpy as np
import os
import sys

path_to_script = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_to_script)

import logging
import filepopulator

# Create your models here

settings.LOGGER.warning("TODO: Implement model for faces")

def face_thumbnail_path(instance, filename):
    first_dir = filename[:2]
    second_dir = filename[2]
    return f"face_thumbnails/{first_dir}/{second_dir}/{filename}"

def face_highlight_path(instance, filename):
    first_dir = filename[:1]
    return f"face_highlights/{first_dir}/{filename}"

def get_default_blank_person():
    
    # Get the blank face name person, or create if they don't
    # exist.
    blank_face_default = Person.objects.get(person_name=settings.BLANK_FACE_NAME)
    if blank_face_default is None:
        new_person = Person(person_name = settings.BLANK_FACE_NAME)
        # Read the blank face image (thanks XKCD). It's saved
        # in the project now. 
        person_thumb = cv2.imread(settings.BLANK_FACE_IMG_PATH)
        personal_thumbnail = cv2.resize(sq_thumb, (500, 500))
        # encode the image
        is_success, person_buff = cv2.imencode(".jpg", personal_thumbnail)
        # Save thumbnail to in-memory file as BytesIO
        person_byte_thumbnail = BytesIO(person_buff)

        # Fun binary filename and such.
        person_thumb_filename = f'033003300DEADBEEF003_BLANK.jpg'
        person_byte_thumbnail.seek(0)
        # Load a ContentFile into the thumbnail field so it gets saved
        new_person.highlight_img.save(person_thumb_filename, ContentFile(person_byte_thumbnail.read())) 

        new_person.save()
        
        blank_face_default = Person.objects.get(person_name = settings.BLANK_FACE_NAME)

    return blank_face_default

class Person(models.Model):
    person_name = models.CharField(max_length=256)
    # I'd rather the highlight be its own image rather than
    # a link to a face in case the face gets deleted or moved.
    highlight_img = models.ImageField(upload_to=face_highlight_path, default=None)

    further_images_unlikely = models.BooleanField(default=False)

    gender = models.CharField(max_length=15, default="unknown")

    # Opt-in toggle (same DB-backed-config shape as CameraGpsBackfillConfig
    # -- frontend-editable, no deploy needed to change) -- when set, any
    # photo with a VALIDATED face for this person is eligible for GPS
    # estimation from that person's own OwnTracks location history
    # (filepopulator/owntracks_gps.py's backfill_camera_gps()), regardless
    # of which camera_model took the photo. A validated appearance means
    # this specific person was confirmed physically present at that
    # moment, which is actually a more direct signal than camera
    # ownership (someone else's phone could have taken the photo).
    # db_default (not just default) -- a stale, already-running worker
    # process that still has the pre-migration field list would otherwise
    # emit an INSERT omitting this column entirely, hitting the exact
    # NOT-NULL-violation incident already documented in CLAUDE.md
    # (face_extraction_timeout_count, 2026-09-18) for the same reason.
    gps_backfill_source = models.BooleanField(default=False, db_default=False)

    # Opt-in, frontend-editable per-person override for classify_unassigned()
    # (face_manager/assign_faces.py) -- when set, sim_99th is computed only
    # against this person's gallery faces within +/- classification_window_
    # days of the QUERY face's own date, instead of the whole gallery.
    # Validated empirically (2026-10-25) as a real, substantial false-
    # positive-rate improvement for people whose gallery spans a wide date
    # range with real cross-era appearance drift (aging, hairstyle/glasses
    # changes) -- e.g. Leslie Williams, David Redd -- but NOT a universal
    # win: tested broadly against the full large-gallery population and
    # found to make FPR flat-to-worse for most people (including huge,
    # healthy Lewis/Wilson family galleries), and actively worse for
    # people whose gallery has a genuinely narrow date span (nothing to
    # gain from windowing when there's no cross-era blending to filter
    # out). No reliable automatic predictor was found across repeated
    # attempts (cohesion alone, span alone, cohesion+span combined all
    # over- or under-triggered against the real population) -- so this
    # is deliberately a per-person, individually-validated toggle, not a
    # formula-driven default. Same db_default reasoning as
    # gps_backfill_source above.
    use_windowed_classification = models.BooleanField(default=False, db_default=False)
    classification_window_days = models.IntegerField(default=182, db_default=182)

    # Opt-in, per-person second-opinion veto gate for classify_unassigned()
    # -- when set, a candidate match for this person that ALREADY passed
    # the normal insightface-based gate (bucket threshold + margin/
    # det_score) must ALSO clear a similarity check against a second,
    # independently-trained embedding model (LVFace-B, a ViT-based face
    # recognition model, github.com/bytedance/LVFace) before being
    # accepted -- a real match fails only one way (insightface accepts,
    # LVFace vetoes), never the reverse (LVFace is never used to accept a
    # candidate insightface itself rejected). Built 2026-10-25 after a
    # real investigation (see CLAUDE.md): Leslie Williams' gallery showed
    # an ~87% false-positive rate on random sampling even after the
    # windowed-classification fix above, and LVFace-B was found, via
    # repeated empirical validation at two sample sizes against her real
    # confirmed false-positive cases, to separate them far more cleanly
    # than insightface alone (known-bad candidates scored 0.40-0.50 under
    # insightface -- comfortably past her threshold -- but only 0.16-0.33
    # under LVFace-B, well below her true-match distribution). LVFace-B
    # was also directly compared against the smaller/larger published
    # variants (S/L) on this same real data and found to be the best of
    # the three -- bigger isn't automatically better here, likely because
    # all published LVFace sizes share the same (smaller) Glint360K
    # training set rather than scaling data with capacity.
    #
    # Deliberately per-person, not a global default, matching
    # use_windowed_classification's own reasoning: this has only been
    # empirically validated for Leslie Williams so far, not the broader
    # population, and real compute cost (LVFace-B inference is far slower
    # than insightface's) makes a global rollout something to earn
    # gallery-by-gallery, not assume. See face_manager/lvface.py for the
    # embedding/veto implementation and LVFACE_VETO_THRESHOLD's own
    # comment for where that constant came from.
    use_lvface_verification = models.BooleanField(default=False, db_default=False)

    # id field has a primary key
    def delete(self):
        # Protect the no face assigned person from
        # ever being deleted.
        if self.person_name == settings.BLANK_FACE_NAME:
            return
        # Else, remove the saved image and
        # then delete the person object.
        try:
            os.remove(self.highlight_img.path)

        except FileNotFoundError:
            pass
        except ValueError as ve:
            if 'has no file associated' in ve.args[0]:
                pass
            else:
                raise ve
        super(Person, self).delete()

    def __str__(self):
        return self.person_name

class SingleFloatField(models.FloatField):
    """Django has no built-in single-precision float field -- FloatField
    always maps to Postgres's double precision. insightface's embeddings
    are natively float32 (verified directly: insight_detected_face
    ['embedding'].dtype), so storing them as double precision only adds
    artificial precision the model never produced (NumPy's .tolist() on a
    float32 array widens every value to a Python float, i.e. a C double,
    with no rounding -- Postgres then stores that double as-is). Widening
    float32->float64 is always exact, so storing at `real` instead loses
    nothing that wasn't already lost by the model itself. Verified against
    634k real production values: 100% exact bit-for-bit round-trip
    through float32, and <1e-7 cosine-similarity distortion on random
    pairs -- far below the 0.6 classification threshold.
    """
    def db_type(self, connection):
        return 'real'


class Face(models.Model):

    # Source of truth for how many poss_identN/weight_N field pairs exist
    # below. face_manager/apps.py registers a system check that fails
    # manage.py check/startup if this ever drifts from the actual fields.
    NUM_POSSIBLE_IDENTITIES = 5

    # Primary key (id) comes for free.
    # For all the foreign keys to person, we set the on_delete method
    # to models.SET. This property takes a function that returns
    # a given object -- i.e. the "_NO_FACE_ASSIGNED_" person.
    # Importantly, the field takes a function rather than
    # an object. See above for the definition of the function.
    declared_name = models.ForeignKey('Person', \
        on_delete=models.SET(get_default_blank_person), related_name='face_declared', \
        blank=True, null=True)
    source_image_file = models.ForeignKey('filepopulator.ImageFile', on_delete=models.CASCADE, blank=True, null=True)
    # Video-derived faces (Phase 3, face_manager/video_face_pipeline.py):
    # one row per final union-merge track-group, not one per detected
    # frame -- see CLAUDE.md's Phase 3 design for the full rationale.
    # CASCADE for the same reason as source_image_file: deleting the
    # source video should delete the faces derived from it. Exactly one
    # of source_image_file/source_video_file must be set (see the
    # CheckConstraint in Meta below).
    source_video_file = models.ForeignKey('filepopulator.VideoFile', on_delete=models.CASCADE, blank=True, null=True)
    # First/last real appearance of this track-group within the video, in
    # seconds. Deliberately not a full per-segment list even though a
    # union-merge group is frequently discontiguous (stitching together
    # separate appearances) -- this pipeline is aimed at "who's likely in
    # this video," not complete per-appearance metadata.
    video_first_timestamp_seconds = models.FloatField(blank=True, null=True)
    video_last_timestamp_seconds = models.FloatField(blank=True, null=True)
    # Exact timestamp of the frame the current thumbnail/box/kps came from
    # -- lets the API re-extract that specific frame on demand (via ffmpeg)
    # to serve a "full context" image for this face, the video equivalent
    # of an image-sourced face's source_image_file.filename. Deliberately
    # NOT storing the extracted frame itself (would cost real disk space
    # across the whole library) -- this field is the only new state needed
    # to make on-demand re-extraction possible.
    video_thumbnail_frame_seconds = models.FloatField(blank=True, null=True)
    # ArrayField supported in PostGres
    dateTakenUTC = models.FloatField(default=0)

    reencoded = models.BooleanField(default=False)

    face_encoding_512 = ArrayField(
                            SingleFloatField(),
                            size=512, blank=True, null=True
                        )

    # LVFace-B (ViT) embedding -- see Person.use_lvface_verification's own
    # docstring for the full investigation. Populated going forward by
    # the live extraction pipeline (face_extract_encode.py, right
    # alongside face_encoding_512 -- same kps-based alignment, no extra
    # detection needed) for EVERY newly-detected face, not just opted-in
    # galleries -- per the user's explicit call (2026-10-25) that having
    # this ready for the whole library is worth the ongoing extraction
    # cost, even though only one person (so far) has the veto gate
    # itself turned on. Existing faces predating this field are covered
    # by the one-time `backfill_lvface_encoding` management command
    # (same resumable/checkpointed shape as backfill_det_score) --
    # expect that to take 2-3 days of continuous single-threaded CPU
    # time for the full ~638k-face library, since LVFace-B inference
    # (~0.3-0.4s/face on CPU) is far more expensive than insightface's
    # own lightweight detector. NULL means "not yet backfilled" (a real,
    # temporary gap during the backfill's run), not "intentionally
    # skipped" -- classify_unassigned() treats a NULL query or gallery
    # embedding here the same way it already treats a NULL det_score:
    # skip the LVFace half of the gate rather than blocking outright.
    face_encoding_lvface_b = ArrayField(
                            SingleFloatField(),
                            size=512, blank=True, null=True
                        )

    # LVFace-S and LVFace-L (the smaller/larger published Glint360K
    # variants, alongside B above) -- added 2026-10-26 to properly
    # re-test which size is actually worth using, after the original
    # B-only validation (see Person.use_lvface_verification's docstring)
    # turned out to rest on a flawed, too-small sample. Same NULL-means-
    # not-yet-backfilled semantics as face_encoding_lvface_b. Populated
    # by the rigorous redetect+bipartite-match+verify backfill (see
    # CLAUDE.md), not the live extraction pipeline -- these two sizes
    # are investigation-only for now, not wired into any live gate.
    face_encoding_lvface_s = ArrayField(
                            SingleFloatField(),
                            size=512, blank=True, null=True
                        )
    face_encoding_lvface_l = ArrayField(
                            SingleFloatField(),
                            size=512, blank=True, null=True
                        )

    # ADDITIVE, dual-write-phase halfvec (float16) mirrors of the 4
    # embedding fields above (2026-10-30 -- see CLAUDE.md's pgvector
    # investigation). Validated empirically before building this,
    # not assumed: real cosine-similarity drift from float16 rounding
    # is ~1e-5 on average, max ~7e-5 even in the classification-relevant
    # 0.3-0.7 similarity band -- three orders of magnitude below this
    # project's tightest real threshold margin (CANDIDATE_MARGIN_FLOOR=
    # 0.03) -- and a real pg_dump/pg_restore spike confirmed backups
    # restore cleanly PROVIDED the target Postgres also has the vector
    # extension built in (now true for both master's Dockerfile_postgres
    # and a rebuilt db_picasa_dev; NOT yet true for the live, still-
    # running db_picasa/picasa_api containers -- deliberately deferred,
    # see the deploy note elsewhere in this file, since multi-day
    # backfill/remediation jobs were actively running in picasa_api at
    # the time these fields were added).
    #
    # These are NOT yet read by anything -- this is purely the additive
    # schema step. Every WRITE call site (face_extract_encode.py,
    # video_face_pipeline.py, the LVFace/verified-kps backfill commands,
    # remediate_video_face_geometry.py, etc.) needs to be updated to
    # dual-write these alongside the existing float32 ArrayField, THEN a
    # one-time backfill converts every existing row, THEN (only once
    # verified against real restored/backfilled data) read call sites
    # cut over one subsystem at a time, THEN -- only after a real bake-in
    # period and a fresh backup -- the old float32 columns get dropped to
    # actually reclaim the storage. Nothing here is a cutover by itself.
    #
    # A real, confirmed ORM-compatibility gap, found via a direct spike
    # (not assumed): exact-equality lookups (`filter(field=value)`,
    # `~Q(field=value)`) work correctly against HalfVectorField -- every
    # one of the 8 real NON_DETECTED_FACE_ENCODING sentinel-equality call
    # sites (assign_faces.py x4, verification_clustering.py,
    # face_extract_encode.py, redetect_sar_affected_videos.py) will work
    # unmodified once cut over. Per-element index lookups (Django's
    # ArrayField `__0` transform, SQL `emb[1]`) do NOT work --
    # HalfVectorField raises a clean FieldError, not a silent wrong
    # result -- so cleanup_chronically_unmatched_faces.py's
    # `face_encoding_512__0=...` check needs a small rewrite (full-vector
    # equality instead of a first-element check) before that file can be
    # cut over; not yet done.
    face_encoding_512_hv = HalfVectorField(dimensions=512, blank=True, null=True)
    face_encoding_lvface_b_hv = HalfVectorField(dimensions=512, blank=True, null=True)
    face_encoding_lvface_s_hv = HalfVectorField(dimensions=512, blank=True, null=True)
    face_encoding_lvface_l_hv = HalfVectorField(dimensions=512, blank=True, null=True)

    # The 5 facial landmark points (left eye, right eye, nose, left mouth
    # corner, right mouth corner) InsightFace's detector produces, flattened
    # to [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5] in the SOURCE IMAGE's pixel
    # coordinate space (not the face's own crop). These are what
    # insightface.utils.face_align.norm_crop() uses to build the aligned
    # 112x112 crop the recognition model actually sees -- with them saved,
    # a face's embedding can be exactly reproduced later (e.g. after
    # face_encoding_512 is cleared to save storage) via a single recognition
    # pass against the original image, with no re-detection needed at all.
    # Nullable since faces detected before this field existed have none;
    # only newly detected/re-matched faces populate it going forward.
    kps = ArrayField(
                SingleFloatField(),
                size=10, blank=True, null=True
            )

    # The detector's own confidence for this detection (insightface's
    # det_score, 0-1) -- NOT previously stored anywhere despite being
    # available on every detection since day one. Added 2026-09-18 after
    # a real investigation found it cleanly separates genuine faces from
    # bogus detections that happen to numerically match a real person's
    # gallery (a mean ~0.85 for validated real faces vs. ~0.58 for
    # confirmed-wrong candidates in a real sample, zero overlap -- see
    # CLAUDE.md). Nullable since faces detected before this field existed
    # have none; populated going forward by find_and_encode_faces(), and
    # backfilled for the existing unlabeled .ignore population via
    # backfill_det_score (management command) using the same re-detect-
    # and-IOU-match technique reencode_missing_faces() already uses for
    # missing embeddings.
    det_score = models.FloatField(null=True, blank=True, default=None)

    # min(box_width, box_height) / min(image_width, image_height) at
    # detection time -- relative, not an absolute pixel count, so one
    # number is comparable across wildly different source resolutions.
    # Stored (not just computed on demand) so a frontend tagging queue
    # can filter cheaply without joining back to the source image's own
    # dimensions on every request. See PhotoCollection.
    # min_face_relative_size (filepopulator/models.py) for the
    # per-collection floor this is compared against at detection time.
    relative_size = models.FloatField(null=True, blank=True, default=None)

    # Populated by the nightly face_manager.cluster_unverified_faces task:
    # per-person complete-linkage clustering (see verification_clustering.py)
    # over this person's own unverified faces, so a human reviewing a run
    # can spot-check a whole visually-coherent group at once. Group ids are
    # 0-indexed PER PERSON (not globally unique) and assigned only to
    # clusters of size >=2 -- a singleton face is left NULL, same as a face
    # that's ineligible for clustering entirely (verified, ignored, no
    # usable embedding). Cleared to NULL immediately whenever a face's
    # assignment changes (associate_person/verify_person_in_image/
    # reset_to_pool) rather than waiting for the next nightly rebuild, and
    # also cleared for every face at the start of each nightly run before
    # re-clustering from scratch -- no attempt is made to preserve group
    # identity night-to-night.
    verification_cluster_group = models.IntegerField(null=True, blank=True, default=None)

    detected_age = models.IntegerField(default=-1)

    # This field will contain the top 5 possible identities as categorized
    # by the FC network.
    # Like the declared name, these also have the on_delete method set. 
    # No need to worry much about this -- these fields are transient
    # by nature and will be reassigned to existing people objects
    # the next time the FC network runs.
    poss_ident1 = models.ForeignKey('Person', on_delete=models.SET(get_default_blank_person), related_name='face_poss1', \
        blank=True, null=True)
    weight_1 = models.FloatField(default=0.0)
    poss_ident2 = models.ForeignKey('Person', on_delete=models.SET(get_default_blank_person), related_name='face_poss2', \
        blank=True, null=True)
    weight_2 = models.FloatField(default=0.0)
    poss_ident3 = models.ForeignKey('Person', on_delete=models.SET(get_default_blank_person), related_name='face_poss3', \
        blank=True, null=True)
    weight_3 = models.FloatField(default=0.0)
    poss_ident4 = models.ForeignKey('Person', on_delete=models.SET(get_default_blank_person), related_name='face_poss4', \
        blank=True, null=True)
    weight_4 = models.FloatField(default=0.0)
    poss_ident5 = models.ForeignKey('Person', on_delete=models.SET(get_default_blank_person), related_name='face_poss5', \
        blank=True, null=True)
    weight_5 = models.FloatField(default=0.0)

    written_to_photo_metadata = models.BooleanField(default=False)

    rejected_fields = ArrayField(
                            models.IntegerField(),
                            size=128, blank=True, null=True
                        )

    # Set True by the PhotoVerify mobile app to hide this face from its
    # review screens without touching the classification (the face stays
    # whatever it was -- unlabeled, a proposed .ignore, etc). Nullable +
    # default None so the ~all rows that are never hidden cost nothing
    # beyond the existing per-row null bitmap.
    mobile_review_hidden = models.BooleanField(null=True, blank=True, default=None)


    # Preserve the values of the face's bounding box.
    box_top = models.IntegerField(validators=[MinValueValidator(1)], default=-1)
    box_bottom = models.IntegerField(validators=[MinValueValidator(1)], default=-1)
    box_left = models.IntegerField(validators=[MinValueValidator(1)], default=-1)
    box_right = models.IntegerField(validators=[MinValueValidator(1)], default=-1)

    # A field to save the thumbnail. The scripts.py ensures
    # that this is a square thumbnail.
    face_thumbnail = models.ImageField(upload_to=face_thumbnail_path, default=None)

    validated = models.BooleanField(default=False)

    detected_gender = models.CharField(max_length=15, default="unknown")
    detected_gender_prob = models.FloatField(default=-1)
    detected_age_group = models.IntegerField(default=-1)
    detected_age_prob = models.FloatField(default=-1)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(source_image_file__isnull=False, source_video_file__isnull=True) |
                    models.Q(source_image_file__isnull=True, source_video_file__isnull=False)
                ),
                name='face_exactly_one_source',
            ),
            # The video_* fields (timestamps, thumbnail-frame timestamp)
            # only ever mean anything for a video-sourced face -- this
            # mirrors the XOR constraint above, guarding against one of
            # them ever being set on an image-sourced row by mistake
            # (e.g. a future code path touching Face without going
            # through video_face_pipeline.py).
            models.CheckConstraint(
                condition=(
                    models.Q(source_video_file__isnull=False) |
                    models.Q(
                        video_first_timestamp_seconds__isnull=True,
                        video_last_timestamp_seconds__isnull=True,
                        video_thumbnail_frame_seconds__isnull=True,
                    )
                ),
                name='face_video_fields_require_video_source',
            ),
            # Structural backstop for a real bug found 2026-10-26 (see
            # CLAUDE.md): PyramidalDetector used to never translate a
            # tiled sub-image detection's kps back to full-image
            # coordinates (only bbox was), silently storing a kps
            # thousands of pixels from its own face's box for ~360k
            # faces in production. Fixed at the source, but the DB
            # itself should refuse to store this class of corruption
            # again regardless of which code path writes it. A 25%
            # margin (not zero) is deliberate -- checked empirically
            # against 2,682 real, presumed-good production faces: a
            # strict zero-margin check already rejects 5.07% of them
            # (real landmark points can legitimately sit just past a
            # tight SCRFD box edge under an extreme pose/expression),
            # while 25% leaves only 0.30% -- nowhere near the
            # thousands-of-pixels scale the real bug produced, so this
            # stays a true backstop against gross corruption rather than
            # a routine blocker on normal detections. NULL kps (not yet
            # backfilled) always passes trivially.
            models.CheckConstraint(
                condition=(
                    models.Q(kps__isnull=True) |
                    models.Q(RawSQL(
                        """
                        kps[1] BETWEEN box_left - (box_right - box_left) * 0.25
                                   AND box_right + (box_right - box_left) * 0.25
                        AND kps[2] BETWEEN box_top - (box_bottom - box_top) * 0.25
                                       AND box_bottom + (box_bottom - box_top) * 0.25
                        AND kps[3] BETWEEN box_left - (box_right - box_left) * 0.25
                                       AND box_right + (box_right - box_left) * 0.25
                        AND kps[4] BETWEEN box_top - (box_bottom - box_top) * 0.25
                                       AND box_bottom + (box_bottom - box_top) * 0.25
                        AND kps[5] BETWEEN box_left - (box_right - box_left) * 0.25
                                       AND box_right + (box_right - box_left) * 0.25
                        AND kps[6] BETWEEN box_top - (box_bottom - box_top) * 0.25
                                       AND box_bottom + (box_bottom - box_top) * 0.25
                        AND kps[7] BETWEEN box_left - (box_right - box_left) * 0.25
                                       AND box_right + (box_right - box_left) * 0.25
                        AND kps[8] BETWEEN box_top - (box_bottom - box_top) * 0.25
                                       AND box_bottom + (box_bottom - box_top) * 0.25
                        AND kps[9] BETWEEN box_left - (box_right - box_left) * 0.25
                                       AND box_right + (box_right - box_left) * 0.25
                        AND kps[10] BETWEEN box_top - (box_bottom - box_top) * 0.25
                                        AND box_bottom + (box_bottom - box_top) * 0.25
                        """,
                        (),
                        output_field=models.BooleanField(),
                    ))
                ),
                name='face_kps_within_box_margin',
            ),
        ]

    def __str__(self):
        return "Face instance of {}".format(self.declared_name)

    def delete(self):
        # Tolerate the thumbnail already being gone from disk -- e.g. a
        # duplicate delete against a restored DB snapshot whose rows
        # point at the same shared media path a live/earlier run already
        # cleaned up. The row itself should still be removed either way.
        try:
            os.remove(self.face_thumbnail.path)
        except FileNotFoundError:
            pass
        super(Face, self).delete()

    def save(self, *args, **kwargs):
        # Basic validation for the bounding box
        if self.box_top >= self.box_bottom:
            raise ValidationError(f'Box bottom ({self.box_bottom}) must be larger value than box top ({self.box_top}) ')
        if self.box_right <= self.box_left:
            raise ValidationError(f'Box right ({self.box_right}) must be larger value than box left ({self.box_left})')
        # Get the source media's height and width -- exactly one of
        # source_image_file/source_video_file is set (see Meta.constraints).
        if self.source_video_file_id is not None:
            img_h = self.source_video_file.height
            img_w = self.source_video_file.width
        else:
            img_h = self.source_image_file.height
            img_w = self.source_image_file.width

        if self.box_top < 0:
            raise ValidationError(f"Box top {self.box_top} is < 0")
        if self.box_left < 0:
            raise ValidationError(f"Box left {self.box_left} is < 0")
        if self.box_bottom < 0:
            raise ValidationError(f"Box bottom {self.box_bottom} is off the edge of the photo {img_h}")
        if self.box_right < 0:
            raise ValidationError(f"Box right {self.box_right} is off the edge of the photo {img_w}")

        # .path (not .file.name) deliberately -- .path resolves the
        # filesystem path via the storage backend with no file I/O at
        # all, while .file.name OPENS the underlying file (via FieldFile.
        # _get_file()) and Django never closes it automatically. A caller
        # doing many saves in one long-lived process (e.g. a backfill)
        # would otherwise accumulate open file handles until the OS fd
        # limit is hit (confirmed real: OSError: Too many open files at
        # ~600 images into backfill_det_score). Safe here specifically
        # because this project's storage backend is always local
        # filesystem (FileSystemStorage) -- .path raises NotImplementedError
        # on a remote/streaming-only backend, which .file.name would not.
        if not os.path.exists(self.face_thumbnail.path):
            raise ValidationError(f"Face thumbnail image does not exist on the OS")

        return super().save(*args, **kwargs)

    def remove_poss_ident(self, poss_idx):
        field_name = f'poss_ident{poss_idx}'
        if not hasattr(self, field_name):
            raise AttributeError(
                f"Face has no field '{field_name}' -- poss_idx must be between "
                f"1 and {self.NUM_POSSIBLE_IDENTITIES} (NUM_POSSIBLE_IDENTITIES)."
            )
        if getattr(self, field_name) is not None:
            # Real attribute assignment (not a raw __dict__/attname poke) so
            # Model.save() doesn't reconcile the still-cached related object
            # back over this on write.
            setattr(self, field_name, None)
            setattr(self, f'weight_{poss_idx}', 0.0)

    def associate_person(self, person_id):
        # A one-stop-shop function to assign a Face to a given
        # Person. Unassociates the face with the old Person, if
        # any. Also removes values for possible identities.

        assert type(person_id) is int, f"person_id must be an int corresponding to a database ID for a Person object."
        assert Person.objects.filter(id=person_id).exists(), f"Person with ID {person_id} does not exist in the database"

        new_id = Person.objects.get(id=person_id)
        self.declared_name = new_id
        self.validated = False
        self.written_to_photo_metadata = False
        self.verification_cluster_group = None

        for poss_idx in range(1, self.NUM_POSSIBLE_IDENTITIES + 1):
            self.remove_poss_ident(poss_idx)

        self.save()

    def verify_person_in_image(self):
        self.validated = True
        self.verification_cluster_group = None
        self.save()

    def reset_to_pool(self, blank=None):
        """Return this face to the unassigned pool for re-classification:
        drop the name assignment and all poss_identN guesses, and set
        declared_name to the blank sentinel Person.

        declared_name is set to the sentinel, NOT NULL: a NULL
        declared_name is invisible both to the "Unassigned" bucket and to
        assign_faces.py's re-classification pass, which each filter on
        declared_name__person_name == settings.BLANK_FACE_NAME. (The old
        clear_person() nulled it, which stranded faces; see the regression
        test in api/tests.py.)

        Pass `blank` to reuse an already-fetched sentinel Person across a
        batch and skip the per-call Person.objects.get().
        """
        if blank is None:
            blank = Person.objects.get(person_name=settings.BLANK_FACE_NAME)

        self.verification_cluster_group = None

        self.set_possibles_zero()  # clears poss_identN + weights, saves

        if self.declared_name_id != blank.id:
            self.declared_name = blank
            self.validated = False
            self.written_to_photo_metadata = False
            self.save()

    def set_possible_person(self, person_id, poss_idx, weight, save=True):
        """save=False lets a caller setting multiple poss_identN slots on
        the same face (assign_faces.py's classify_unassigned() does up to
        5, once per ranked candidate) batch them into a single .save()
        instead of one per call -- Face.save() does real validation work
        and isn't cheap (~20ms measured against production), so 5 calls
        each saving was a real, avoidable cost multiplier. Callers that
        pass save=False are responsible for calling .save() themselves
        once they're done setting fields.

        person_id may be a Person instance instead of a raw id -- lets a
        caller that's already cached the Person objects it needs (e.g.
        faceAssigner.person_cache, avoiding a Person.objects.get() round
        trip on every one of these calls across a large reprocess) pass
        it straight through."""

        assert poss_idx > 0, \
            f'The index correlated to poss_ident must be a value between 1 and {self.NUM_POSSIBLE_IDENTITIES}.'
        assert poss_idx <= self.NUM_POSSIBLE_IDENTITIES, \
            f'The index correlated to poss_ident must be a value between 1 and {self.NUM_POSSIBLE_IDENTITIES}.'
        assert isinstance(person_id, (int, np.int32, np.int64, Person)), \
            f"Person ID should be an int or Person but is {type(person_id)}"
        assert type(weight) in [int, float, np.float64], f"Weight should be an int or float; is {type(weight)}"
        assert weight >= 0
        assert weight <= 1.000001, f"weight was {weight}"
        new_poss_id = person_id if isinstance(person_id, Person) else Person.objects.get(id=person_id)

        setattr(self, f'poss_ident{poss_idx}', new_poss_id)
        setattr(self, f'weight_{poss_idx}', weight)

        if save:
            self.save()

    def set_possibles_zero(self):
        for poss_idx in range(1, self.NUM_POSSIBLE_IDENTITIES + 1):
            self.remove_poss_ident(poss_idx)

        self.save()


    def reject_association(self, person_unassociate_id):

        assert type(person_unassociate_id) == int
        disown_person = Person.objects.get(id=person_unassociate_id)

        possible_ids = []
        for ID_num in range(1, self.NUM_POSSIBLE_IDENTITIES + 1):
            candidate = getattr(self, f'poss_ident{ID_num}')
            possible_ids.append(candidate.id if candidate is not None else None)

        assert person_unassociate_id in possible_ids

        # Find if the removal ID is in the possible IDs, then bump everything
        # up higher in the possible IDs list, accounting for any "None" values.
        if person_unassociate_id in possible_ids:
            remove_idx = possible_ids.index(person_unassociate_id)
            source_idcs = [x for x in range(self.NUM_POSSIBLE_IDENTITIES) if x != remove_idx and possible_ids[x] is not None]
            dest_idcs = list(range(len(source_idcs)))


            for offset in range(len(source_idcs)):
                source_offset = source_idcs[offset]
                dest_offset = dest_idcs[offset]
                if source_offset == dest_offset:
                    continue
                setattr(self, f'poss_ident{dest_offset + 1}', getattr(self, f'poss_ident{source_offset + 1}'))
                setattr(self, f'weight_{dest_offset + 1}', getattr(self, f'weight_{source_offset + 1}'))

            if len(dest_idcs) > 0:
                offset_start = max(dest_idcs) + 2
            else:
                offset_start = 1

            for offset in range(offset_start, self.NUM_POSSIBLE_IDENTITIES + 1):
                setattr(self, f'poss_ident{offset}', None)
                setattr(self, f'weight_{offset}', 0.0)

        self.add_to_rejected_fields(person_unassociate_id)
        self.save()

    def add_to_rejected_fields(self, person_id):
        """Record that `person_id` has been explicitly ruled out for this
        face -- classify_unassigned() checks this list so it doesn't just
        re-propose a candidate a human already declined. Does NOT save;
        callers that also make other changes in the same operation (e.g.
        close_assigned's "Remove from person" case, which immediately
        calls associate_person()) can set this first and let that other
        call's own save() persist both changes together."""
        reject_list = self.rejected_fields
        if reject_list is None:
            reject_list = []
        reject_list.append(person_id)
        self.rejected_fields = list(set(reject_list))


def clear_confirmed_ignore_face_encodings():
    """Null face_encoding_512 for every face CONFIRMED (declared_name) to
    .ignore/.realignore, leaving Face.kps untouched so
    FaceExtractor.reencode_missing_faces() can still recover a usable
    embedding later if the face is ever reassigned away from that
    sentinel. Never touches a face merely SUGGESTED as ignore
    (poss_ident1 only, declared_name still the blank sentinel) --
    classify_unassigned() only ever writes suggestions there, so
    declared_name reaching .ignore/.realignore always means a human
    confirmed it via associate_person(). Single source of truth for the
    actual write, shared by the clear_confirmed_ignore_encodings
    management command (one-off/backfill use) and the
    face_manager.clear_ignored_encodings Celery task (ongoing use, for
    faces confirmed after the backfill). Returns the number of faces
    updated. Safe to call repeatedly -- matches nothing once already
    cleared."""
    qs = Face.objects.filter(
        declared_name__person_name__in=[settings.SOFT_IGNORE_NAME, '.realignore'],
    ).exclude(face_encoding_512__isnull=True)
    return qs.update(face_encoding_512=None)


def clear_encodings_below_collection_floor(collection=None, dry_run=False):
    """Apply a PhotoCollection's CURRENT min_face_det_score/
    min_face_relative_size floor to faces that already have a usable
    embedding -- add_new_face()/FaceExtractor.reencode_missing_faces()
    only ever gate a face that DOESN'T already have one, so retuning a
    floor down has no effect on the existing backlog without this.

    Deliberately scoped to faces never human-adjudicated
    (declared_name == BLANK_FACE_NAME, validated=False) -- nothing a
    human has confirmed or verified is ever stripped of its embedding,
    regardless of how small/low-confidence the detection looks. The
    Face row, box, and thumbnail are untouched either way; only the
    embedding (and therefore future gallery-comparison participation)
    is withheld.

    Also clears every poss_identN/weight_N on the face (same shape as
    remove_poss_ident(), without the per-row save() that method does) --
    a stale proposal whose backing embedding no longer exists is both
    misleading (a face shown as "proposed as X" with nothing left to
    justify it) and a real crash risk (classify_unassigned() assumes a
    usable embedding on anything it's handed; a targeted reclassification
    pass that doesn't separately filter this out raises a TypeError on
    np.linalg.norm(None), confirmed in production 2026-10-20). A
    confirmed (declared_name/validated) face already gets this via
    associate_person()/verify_person_in_image() at confirmation time --
    this function's own scope (never-adjudicated faces only) is exactly
    the population that wouldn't otherwise go through that path.

    collection=None sweeps every image-backed, not-yet-adjudicated face
    in the whole library (the one-off/manual-backfill shape); passing a
    specific PhotoCollection scopes the sweep to just that collection's
    images, which is all the automatic save()-triggered re-application
    (see PhotoCollection.save()) ever needs.

    Returns (scanned, cleared). Single source of truth for the actual
    write, shared by the clear_encodings_below_floor management command
    and the face_manager.apply_collection_floor_change Celery task."""
    from filepopulator.models import below_collection_floor

    BATCH_SIZE = 5000
    update_fields = ['face_encoding_512']
    for poss_idx in range(1, Face.NUM_POSSIBLE_IDENTITIES + 1):
        update_fields.append(f'poss_ident{poss_idx}')
        update_fields.append(f'weight_{poss_idx}')

    candidates = Face.objects.filter(
        declared_name__person_name=settings.BLANK_FACE_NAME,
        validated=False,
        face_encoding_512__isnull=False,
        source_image_file__isnull=False,
    ).exclude(
        face_encoding_512=settings.NON_DETECTED_FACE_ENCODING,
    ).select_related('source_image_file__directory__collection')
    if collection is not None:
        candidates = candidates.filter(source_image_file__directory__collection=collection)

    to_update = []
    scanned = 0
    cleared = 0
    for face in candidates.iterator(chunk_size=BATCH_SIZE):
        scanned += 1
        img_obj = face.source_image_file
        if not below_collection_floor(img_obj.directory.collection, face.det_score, face.relative_size):
            continue
        cleared += 1
        if not dry_run:
            face.face_encoding_512 = None
            for poss_idx in range(1, Face.NUM_POSSIBLE_IDENTITIES + 1):
                setattr(face, f'poss_ident{poss_idx}', None)
                setattr(face, f'weight_{poss_idx}', 0.0)
            to_update.append(face)
            if len(to_update) >= BATCH_SIZE:
                Face.objects.bulk_update(to_update, update_fields, batch_size=BATCH_SIZE)
                to_update = []

    if not dry_run and to_update:
        Face.objects.bulk_update(to_update, update_fields, batch_size=BATCH_SIZE)

    return scanned, cleared


def clear_stale_proposals_with_no_encoding(collection=None, dry_run=False):
    """Catch-up for the backlog clear_encodings_below_collection_floor()
    couldn't reach: that function only clears poss_identN/weight_N on a
    face it ITSELF nulls the encoding on -- a face whose encoding was
    already NULL before this fix existed (e.g. 32,344 faces cleared by
    the one-time library-wide sweep run 2026-10-20, before this function
    was added) is invisible to it (its own candidate query requires
    face_encoding_512__isnull=False, so an already-NULL row never
    matches). This sweeps the opposite condition: any never-adjudicated
    face (declared_name == BLANK_FACE_NAME, validated=False) that's
    missing an encoding but still carries a stale poss_identN proposal,
    regardless of why the encoding went missing.

    A plain queryset .update() (no bulk_update needed -- every row gets
    the same static values, not a per-row Python-computed one), so this
    is cheap even at full-library scale.

    collection=None sweeps the whole library; passing a specific
    PhotoCollection scopes it to that collection's images only -- used
    by face_manager.apply_collection_floor_change so a threshold change
    also mops up any stale backlog for that collection on every run, not
    just newly-cleared faces.

    Returns the number of faces updated."""
    candidates = Face.objects.filter(
        declared_name__person_name=settings.BLANK_FACE_NAME,
        validated=False,
        face_encoding_512__isnull=True,
        source_image_file__isnull=False,
    )
    if collection is not None:
        candidates = candidates.filter(source_image_file__directory__collection=collection)

    has_any_proposal = Q()
    for poss_idx in range(1, Face.NUM_POSSIBLE_IDENTITIES + 1):
        has_any_proposal |= Q(**{f'poss_ident{poss_idx}__isnull': False})
    candidates = candidates.filter(has_any_proposal)

    if dry_run:
        return candidates.count()

    update_kwargs = {}
    for poss_idx in range(1, Face.NUM_POSSIBLE_IDENTITIES + 1):
        update_kwargs[f'poss_ident{poss_idx}'] = None
        update_kwargs[f'weight_{poss_idx}'] = 0.0
    return candidates.update(**update_kwargs)

