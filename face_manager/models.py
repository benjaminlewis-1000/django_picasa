from django.db import models
from django.conf import settings
from django.forms import ModelForm
from django.core.validators import MinValueValidator, MaxValueValidator, RegexValidator
from django.core.validators import *
from datetime import datetime
from django.utils import timezone
from django.contrib.postgres.fields import ArrayField
import cv2

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


    def remove_poss_ident(self, poss_idx):
        if self.__dict__[f'poss_ident{poss_idx}_id'] != None:
            ident_id_person = self.__dict__[f'poss_ident{poss_idx}_id']
            person = Person.objects.get(id=ident_id_person)
            person.decrement_possible_num()
            self.__dict__[f'poss_ident{poss_idx}_id'] = None
            self.__dict__[f'weight_{poss_idx}'] = 0.0

    def associate_person(self, person_id):
        # A one-stop-shop function to assign a Face to a given
        # Person. Unassociates the face with the old Person, if
        # any, and decrements the counts of assigned faces. 
        # Also removes values for possible identities. Increments
        # counts for the new person appropriately.

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

        new_poss_id = Person.objects.get(id=person_id)
        if poss_idx == 1:
            new_poss_id.increment_possible_num()
        self.__dict__[f'weight_{poss_idx}'] = weight

        if poss_idx == 1:
            self.poss_ident1 = new_poss_id
        elif poss_idx == 2:
            self.poss_ident2 = new_poss_id
        elif poss_idx == 3:
            self.poss_ident3 = new_poss_id
        elif poss_idx == 4:
            self.poss_ident4 = new_poss_id
        elif poss_idx == 5:
            self.poss_ident5 = new_poss_id

        self.save()

    def set_possibles_zero(self):
        self.remove_poss_ident(1)
        self.remove_poss_ident(2)
        self.remove_poss_ident(3)
        self.remove_poss_ident(4)
        self.remove_poss_ident(5)

        self.save()


    def reject_association(self, person_unassociate_id):

        disown_person = Person.objects.get(id=person_unassociate_id)

        if self.poss_ident1 == disown_person:
            self.poss_ident1.decrement_possible_num()
            self.poss_ident1 = None
            self.weight_1 = 0.0
        elif self.poss_ident2 == disown_person:
            self.poss_ident2.decrement_possible_num()
            self.poss_ident2 = None
            self.weight_2 = 0.0
        elif self.poss_ident3 == disown_person:
            self.poss_ident3.decrement_possible_num()
            self.poss_ident3 = None
            self.weight_3 = 0.0
        elif self.poss_ident4 == disown_person:
            self.poss_ident4.decrement_possible_num()
            self.poss_ident4 = None
            self.weight_4 = 0.0
        elif self.poss_ident5 == disown_person:
            self.poss_ident5.decrement_possible_num()
            self.poss_ident5 = None
            self.weight_5 = 0.0

        reject_list = self.rejected_fields
        if reject_list is None:
            reject_list = []

        reject_list.append(person_unassociate_id)
        # Remove duplicates
        reject_list = list(set(reject_list))

        self.rejected_fields = reject_list
        self.save()

'''
from face_manager.models import Person
people = Person.objects.all()
for p in people:
    print(p)
    p.num_faces = p.face_declared.count()
    p.num_possibilities = p.face_poss1.count() + p.face_poss2.count() + p.face_poss3.count()+ p.face_poss4.count()+ p.face_poss5.count()
    p.num_unverified_faces = p.face_declared.filter(validated=False).count()
    p.save()
'''
