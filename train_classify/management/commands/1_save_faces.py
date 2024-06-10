#! /usr/bin/env python

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count
from django.db.models import Q
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from itertools import chain
from PIL import Image, ExifTags
from random import randint
import cv2
import numpy as np
import os
import pickle
import PIL
import shutil
import time
import face_recognition

class Command(BaseCommand):

    def __init__(self):
        super(Command, self).__init__()

        self.max_dim = 600

        self.out_dir = '/photos_rw/pkls'
        # Get only people with a minimum of 50 images. 
        self.min_faces = 50

        img_path = '/photos/Completed/Pictures_finished/Family Pictures/2017/Mom Phone/1478528587722.jpg'
        # self.test_img(img_path)
        # exit()

    def test_img(self, img_path):
        print("IMG PATH", img_path)
        # Get the file object
        file = ImageFile.objects.filter( filename = img_path)
        faces = Face.objects.filter(source_image_file=file[0])
        face = faces[0]
        print(file, faces)

        data = self.extract_chip(face)
        chip = data['chip']
        bbox = data["bbox"]
        
        chip = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)

        image = face.source_image_file.filename
        image = self.open_img_oriented(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imwrite('/code/tmp_data/chip.png', chip)
        cv2.imwrite('/code/tmp_data/image.png', image)
        print(bbox)


    def get_image_lists(self):
        # Loop over all named faces, and then over ignored/non-interested
        # people images. 


        names = Person.objects.annotate(c=Count('face_declared')) \
            .filter(c__gt=self.min_faces) \
            .filter(~Q(person_name__in=settings.IGNORED_NAMES) ) 

        self.names_list = list(names)

        self.in_library_images = Face.objects.filter(declared_name__in=names).order_by('?')

        criterion_rejected = Q( declared_name__person_name__in=['.ignore', '.realignore'])
        self.ignore_images = Face.objects.filter(criterion_rejected).order_by('?')

    def open_img_oriented(self, filename):

        try:
            image = PIL.Image.open(filename)
        except Exception as e:
            print("EX", e)
            return None

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break

        try:
            exif=dict(image._getexif().items())
        except Exception as e:
            exif = {}

        if orientation in exif.keys():
            if exif[orientation] == 3:
                image=image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image=image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image=image.rotate(90, expand=True)

        # print(image.shape)
        image = np.array(image)
        return image
        

    def extract_chip(self, face):
        # Get a chip with a 25% margin on all sides, 
        # since that is the margin that face_recognition uses to 
        # compute the face encodings. We get the face's image, 
        # chip it out with the margin, and adjust the bounding
        # box to be relative to that chip.

        # Note that the order of face_recognition bounding boxes
        # is [top, right, bottom, left]. 

        l = face.box_left
        r = face.box_right
        b = face.box_bottom
        t = face.box_top

        # Load image to np array
        image = face.source_image_file.filename
        image = self.open_img_oriented(image) # face_recognition.load_image_file(image)
        imh, imw = image.shape[:2]

        if l < 0:
            l = 0
        if t < 0:
            t = 0
        if r > imw:
            r = imw
        if b > imh:
            b = imh
        if r < 0 or b < 0 or t > imh or l > imw:
            return False

        assert l >= 0
        assert r >= 0
        assert b >= 0
        assert t >= 0
        assert l <= imw
        assert r <= imw
        assert b <= imh
        assert t <= imh

        # Compute new image chip
        bbox_width = np.abs(r - l)
        bbox_ht = np.abs(t - b)
        assert bbox_width > 0
        assert bbox_ht > 0

        bbox_pad = 0.25

        t_adj = int(np.max((t - bbox_ht * bbox_pad, 0)))
        b_adj = int(np.min((b + bbox_ht * bbox_pad, imh)))
        l_adj = int(np.max((0, l - bbox_width * bbox_pad)))
        r_adj = int(np.min((imw, r + bbox_width * bbox_pad)))
        assert t_adj <= t, f"{t_adj}, {t} | {t} {r} {b} {l} {imh} {imw}"
        assert l_adj <= l, f"{l_adj}, {l} | {t} {r} {b} {l} {imh} {imw}"
        assert r_adj >= r, f"{r_adj}, {r} | {t} {r} {b} {l} {imh} {imw}"
        assert b_adj >= b, f"{b_adj}, {b} | {t} {r} {b} {l} {imh} {imw}"

        # Chip out the image. 

        bbox_new = [(t - t_adj, l - l_adj + (r - l), t - t_adj + (b - t), l - l_adj)]
        t_diff = np.abs(bbox_new[0][0] - np.abs(t_adj - b_adj) / 6 )
        r_diff = bbox_new[0][1] - np.abs( (r_adj - l_adj - np.abs(r_adj - l_adj) / 6 )) 
        b_diff = bbox_new[0][2] - np.abs( (b_adj - t_adj - np.abs(t_adj - b_adj) / 6 ))
        l_diff = np.abs(bbox_new[0][3] - np.abs(r_adj - l_adj) / 6 )
        
        if t_adj != 0 and b_adj != imh:
            assert t_diff < 1, f'{bbox_new[0][0]} - np.abs({t_adj} - {b_adj}) / 6) | {t}, {r}, {b}, {l}'
            assert b_diff < 1, f'{bbox_new[0][2]} - np.abs( ({b_adj} - {t_adj} - np.abs({t_adj} - {b_adj}) / 6 )) | {t}, {r}, {b}, {l}'
        if l_adj != 0 and r_adj != imw:
            assert r_diff < 1, f'{bbox_new[0][1]} - np.abs( ({r_adj} - {l_adj} - np.abs({r_adj} - {l_adj}) / 6 )) | {t}, {r}, {b}, {l}'
            assert l_diff < 1, f'np.abs({bbox_new[0][3]} - np.abs({r_adj} - {l_adj}) / 6 ) | {t}, {r}, {b}, {l}'
        
        # New image chip: 
        chip = image[t_adj:b_adj, l_adj:r_adj, :]
        assert chip.shape[0] == b_adj - t_adj
        assert chip.shape[1] == r_adj - l_adj

        # RESIZING Function: If the chip is huge, it wastes storage space.
        # This will resize to a manageable size.

        ch, cw = chip.shape[:2]
        if (cw > self.max_dim or ch > self.max_dim):
            rescale = self.max_dim / np.max((ch, cw))

            new_h = int(ch * rescale)
            new_w = int(cw * rescale)

            chip = cv2.resize(chip, dsize=(new_h, new_w), interpolation=cv2.INTER_CUBIC)

            # Rescale the bounding box
            bbox_new = [int(f * rescale) for f in bbox_new[0]]
            bbox_new = [tuple(bbox_new)]

        ch, cw = chip.shape[:2]
        assert ch <= self.max_dim
        assert cw <= self.max_dim

        return {'chip': chip,
                'bbox': bbox_new}


    def img_to_pkl(self, face, set_ign_idx = False):

        data = {}
        data['person_name'] = face.declared_name.person_name
        if not set_ign_idx:
            data['index'] = self.names_list.index(face.declared_name)
        else:
            data['index'] = -999


        if not set_ign_idx:
            out_file = os.path.join(self.out_dir, f"imchip_{face.id}_{data['index']}.pkl")
        else:
            out_file = os.path.join(self.out_dir, f'imchip_ign_{face.id}.pkl')

        if os.path.exists(out_file):
            # print(f"File {out_file} already good")
            return
            
        chip_data = self.extract_chip(face)
        if type(chip_data) is bool:
            return

        data['chip'] = chip_data['chip']
        data['bbox'] = chip_data['bbox']

        data['face_id'] = face.id
        data['img_file'] = face.source_image_file.filename
        data['date_taken'] = face.source_image_file.dateTaken
        data['date_modified'] = face.source_image_file.dateModified

        with open(out_file, 'wb') as fh:
            pickle.dump(data, fh)


    def handle(self, *args, **options):

        # Get the list of in-library and out-of-library images.
        self.get_image_lists()

        n_in_lib = self.in_library_images.count()
        for ii, face in enumerate(self.in_library_images.iterator()):
            try:
                self.img_to_pkl(face, set_ign_idx = False)
            except:
                pass
            if ii % 100 == 0:
                print(f'{ii} / {n_in_lib}, {ii / n_in_lib * 100:.3f}%')
            
        n_out_lib = self.ignore_images.count()
        for ii, face in enumerate(self.ignore_images.iterator()):
            try:
                self.img_to_pkl(face, set_ign_idx = True)
            except:
                pass
            if ii % 100 == 0:
                print(f'{ii} / {n_out_lib}, {ii / n_out_lib * 100:.3f}%')
