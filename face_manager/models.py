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
        self.num_possibilities += 1
        self.save()

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

class Face(models.Model):
    
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
    face_encoding = ArrayField(
                            models.FloatField(),
                            size=128, blank=True, null=True
                        )
    
    face_encoding_512 = ArrayField(
                            models.FloatField(),
                            size=512, blank=True, null=True
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
        os.remove(self.face_thumbnail.path)
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
        if self.__dict__[f'poss_ident{poss_idx}_id'] != None:
            ident_id_person = self.__dict__[f'poss_ident{poss_idx}_id']
            person = Person.objects.get(id=ident_id_person)
            if poss_idx == 1:
                person.decrement_possible_num()
            self.__dict__[f'poss_ident{poss_idx}_id'] = None
            self.__dict__[f'weight_{poss_idx}'] = 0.0

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

        self.remove_poss_ident(1)
        self.remove_poss_ident(2)
        self.remove_poss_ident(3)
        self.remove_poss_ident(4)
        self.remove_poss_ident(5)

        self.save()

    def clear_person(self):
        self.set_possibles_zero()
    
        if self.declared_name != None:
            self.declared_name.decrement_assigned()
            self.declared_name = None
    
        self.save() 

    def verify_person_in_image(self):
        self.declared_name.decrement_unverified()

        self.validated = True
        self.save()

    def set_possible_person(self, person_id, poss_idx, weight):

        assert poss_idx > 0, 'The index correlateed to poss_ident must be a value between 1 and 5.'
        assert poss_idx <= 5, 'The index correlateed to poss_ident must be a value between 1 and 5.'
        assert type(person_id) in [int, np.int32, np.int64], f"Person ID should be an int but is {type(person_id)}"
        assert type(weight) in [int, float, np.float64], f"Weight should be an int or float; is {type(weight)}"
        assert weight >= 0
        assert weight <= 1.000001, f"weight was {weight}"
        new_poss_id = Person.objects.get(id=person_id)
        if poss_idx == 1:
            new_poss_id.increment_possible_num()
        # self.__dict__[f'weight_{poss_idx}'] = weight

        exec(f"self.poss_ident{poss_idx} = new_poss_id")
        exec(f"self.weight_{poss_idx} = weight")

        self.save()

    def set_possibles_zero(self):
        self.remove_poss_ident(1)
        self.remove_poss_ident(2)
        self.remove_poss_ident(3)
        self.remove_poss_ident(4)
        self.remove_poss_ident(5)

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

