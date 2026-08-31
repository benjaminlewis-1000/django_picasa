from django.db import models
from django.conf import settings
from django.forms import ModelForm
from django.core.validators import MinValueValidator, MaxValueValidator, RegexValidator
from django.core.exceptions import ValidationError
from django.core.validators import *
from datetime import datetime
from django.utils import timezone
from django.contrib.postgres.fields import ArrayField
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

    num_faces = models.IntegerField(default=0)
    num_possibilities = models.IntegerField(default=0)
    num_unverified_faces = models.IntegerField(default=0)

    gender = models.CharField(max_length=15, default="unknown")

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

    def increment_assigned(self):
        self.num_faces += 1
        self.save()

    def increment_unverified(self):
        self.num_unverified_faces += 1
        self.save()

    def increment_possible_num(self):
        # Atomic DB-side increment, not a Python-level read-modify-write:
        # faceAssigner caches Person instances across many
        # classify_unassigned() calls (see person_cache in
        # assign_faces.py) to avoid a Person.objects.get() round trip
        # per call, and a multi-threaded reprocess can genuinely call
        # this concurrently on the SAME cached instance for a popular
        # person -- a plain self.num_possibilities += 1; self.save()
        # would lose increments under that race. F() makes the increment
        # itself race-free; refresh_from_db() then syncs this in-memory
        # instance (shared across threads) back to the true DB value
        # rather than leaving it holding a stale local count.
        Person.objects.filter(pk=self.pk).update(num_possibilities=models.F('num_possibilities') + 1)
        self.refresh_from_db(fields=['num_possibilities'])

    def decrement_assigned(self):
        self.num_faces -= 1
        if self.num_faces < 0:
            self.num_faces = 0
        self.save()

    def decrement_unverified(self):
        self.num_unverified_faces -= 1
        if self.num_unverified_faces < 0:
            self.num_unverified_faces = 0
        self.save()

    def decrement_possible_num(self):
        self.num_possibilities -= 1
        if self.num_possibilities < 0:
            self.num_possibilities = 0
        self.save()

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
    # ArrayField supported in PostGres
    dateTakenUTC = models.FloatField(default=0)

    reencoded = models.BooleanField(default=False)

    face_encoding_512 = ArrayField(
                            SingleFloatField(),
                            size=512, blank=True, null=True
                        )

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
        # Get the image height and width
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

        if not os.path.exists(self.face_thumbnail.file.name):
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
            person = getattr(self, field_name)
            if poss_idx == 1:
                person.decrement_possible_num()
            # Real attribute assignment (not a raw __dict__/attname poke) so
            # Model.save() doesn't reconcile the still-cached related object
            # back over this on write.
            setattr(self, field_name, None)
            setattr(self, f'weight_{poss_idx}', 0.0)

    def associate_person(self, person_id):
        # A one-stop-shop function to assign a Face to a given
        # Person. Unassociates the face with the old Person, if
        # any, and decrements the counts of assigned faces. 
        # Also removes values for possible identities. Increments
        # counts for the new person appropriately.

        assert type(person_id) is int, f"person_id must be an int corresponding to a database ID for a Person object."
        assert Person.objects.filter(id=person_id).exists(), f"Person with ID {person_id} does not exist in the database"

        # Change the numbers for assigned to the old and new person objects
        if self.declared_name.id != person_id:
            self.declared_name.decrement_assigned()
            if not self.validated: 
                self.declared_name.decrement_unverified()

        new_id = Person.objects.get(id=person_id)
        self.declared_name = new_id
        new_id.increment_assigned()
        new_id.increment_unverified()
        self.validated = False
        self.written_to_photo_metadata = False

        for poss_idx in range(1, self.NUM_POSSIBLE_IDENTITIES + 1):
            self.remove_poss_ident(poss_idx)

        self.save()

    def verify_person_in_image(self):
        self.declared_name.decrement_unverified()

        self.validated = True
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

        prev = self.declared_name
        if prev is not None and prev.person_name != settings.BLANK_FACE_NAME:
            prev.decrement_assigned()
            if not self.validated:
                prev.decrement_unverified()

        self.set_possibles_zero()  # clears poss_identN + weights, saves

        if self.declared_name_id != blank.id:
            self.declared_name = blank
            self.validated = False
            self.written_to_photo_metadata = False
            blank.increment_assigned()
            blank.increment_unverified()
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

        assert poss_idx > 0, 'The index correlateed to poss_ident must be a value between 1 and 5.'
        assert poss_idx <= 5, 'The index correlateed to poss_ident must be a value between 1 and 5.'
        assert isinstance(person_id, (int, np.int32, np.int64, Person)), \
            f"Person ID should be an int or Person but is {type(person_id)}"
        assert type(weight) in [int, float, np.float64], f"Weight should be an int or float; is {type(weight)}"
        assert weight >= 0
        assert weight <= 1.000001, f"weight was {weight}"
        new_poss_id = person_id if isinstance(person_id, Person) else Person.objects.get(id=person_id)
        if poss_idx == 1:
            new_poss_id.increment_possible_num()
        # self.__dict__[f'weight_{poss_idx}'] = weight

        exec(f"self.poss_ident{poss_idx} = new_poss_id")
        exec(f"self.weight_{poss_idx} = weight")

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
        for ID_num in range(1, 6):
            try:
                get_id = eval(f"self.poss_ident{ID_num}.id")
            except AttributeError:
                get_id = None
            possible_ids.append(get_id)

        assert person_unassociate_id in possible_ids

        # Use eval statements to effect a change in the possible ID list.
        # Find if the removal ID is in the possible IDs, then bump everything
        # up higher in the possible IDs list, accounting for any "None" values. 
        if person_unassociate_id in possible_ids: 
            remove_idx = possible_ids.index(person_unassociate_id)
            exec(f"self.poss_ident{remove_idx + 1}.decrement_possible_num()")
            source_idcs = [x for x in range(5) if x != remove_idx and possible_ids[x] is not None]
            dest_idcs = list(range(len(source_idcs)))


            for offset in range(len(source_idcs)):
                source_offset = source_idcs[offset]
                dest_offset = dest_idcs[offset]
                if source_offset == dest_offset:
                    continue
                exec(f"self.poss_ident{dest_offset + 1} = self.poss_ident{source_offset + 1}")
                exec(f"self.weight_{dest_offset + 1} = self.weight_{source_offset + 1}")
                if dest_offset == 0:
                    # print("Dest offset", self.poss_ident1.person_name, self.poss_ident1.id)
                    self.poss_ident1.increment_possible_num()

            if len(dest_idcs) > 0:
                offset_start = max(dest_idcs) + 2
            else:
                offset_start = 1
                
            for offset in range(offset_start, 6):
                exec(f"self.poss_ident{offset} = None")
                exec(f"self.weight_{offset} = 0.0")

            # print("Removing", remove_idx)
            # if remove_idx == 0:
            #     assert source_idcs[0] == 1
            #     assert dest_idcs[0] == 0

        reject_list = self.rejected_fields
        if reject_list is None:
            reject_list = []

        reject_list.append(person_unassociate_id)
        # Remove duplicates
        reject_list = list(set(reject_list))

        self.rejected_fields = reject_list
        self.save()

