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

        self.max_dim = 800
        self.bbox_pad = 0.4 

        self.out_dir = '/photos_rw/pkls'

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        # Get only people with a minimum of 20 images. 
        self.min_faces = 20

        self.test = False

        img_path = '/photos/Completed/Pictures_finished/Family Pictures/2017/Mom Phone/1478528587722.jpg'
        img_path = '/photos/Completed/Pictures_finished/Family Pictures/2011/2011 (7) July/DSC_0364.JPG'
        if self.test:
            self.test_img(img_path)
        # exit()

    def test_img(self, img_path):
        print("IMG PATH", img_path)
        # Get the file object
        file = ImageFile.objects.filter( filename = img_path)
        faces = Face.objects.filter(source_image_file=file[0])
        face = faces[0]
        print(file, faces)

        data = self.extract_chip(face)
        
        if self.test:
            self.names_list = [face.declared_name]
            self.img_to_pkl(face)
            del self.names_list

        chip = data['chip']
        bbox = data["bbox_trbl"]
        
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

        if self.test:
            print("Testing!!")
            names = names[:10] 

        self.names_list = list(names)

        if self.test:
            self.in_library_images = Face.objects.filter(declared_name__in=names)[:10] 
        else:
            self.in_library_images = Face.objects.filter(declared_name__in=names).order_by('?')


        criterion_rejected = Q( declared_name__person_name__in=['.ignore', '.realignore'])

        if self.test:
            self.ignore_images = Face.objects.filter(criterion_rejected)[:10] 
        else:
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


        t_adj = int(np.max((t - bbox_ht * self.bbox_pad, 0)))
        b_adj = int(np.min((b + bbox_ht * self.bbox_pad, imh)))
        l_adj = int(np.max((0, l - bbox_width * self.bbox_pad)))
        r_adj = int(np.min((imw, r + bbox_width * self.bbox_pad)))
        assert t_adj <= t, f"{t_adj}, {t} | {t} {r} {b} {l} {imh} {imw}"
        assert l_adj <= l, f"{l_adj}, {l} | {t} {r} {b} {l} {imh} {imw}"
        assert r_adj >= r, f"{r_adj}, {r} | {t} {r} {b} {l} {imh} {imw}"
        assert b_adj >= b, f"{b_adj}, {b} | {t} {r} {b} {l} {imh} {imw}"



        bbox_new_t = t - t_adj
        bbox_new_b = b - t_adj # == t - t_adj + (b - t)
        bbox_new_l = l - l_adj
        bbox_new_r = r - l_adj # == l - l_adj + (r - l)

        ## BBOX defined as top, right, bottom, left.
        bbox_new = [(bbox_new_t, bbox_new_r, bbox_new_b, bbox_new_l)]

        # Check assertion that new bounding box is same width
        # and height
        adj_bbox_w = np.abs(bbox_new_l - bbox_new_r)
        adj_bbox_h = np.abs(bbox_new_t - bbox_new_b)
        assert adj_bbox_w == bbox_width, f"Old width: {bbox_width}, new width: {adj_w}"
        assert adj_bbox_h == bbox_ht, f"Old height: {bbox_ht}, new height: {adj_h}"

        # New image chip: 
        chip = image[t_adj:b_adj, l_adj:r_adj, :]
        assert chip.shape[0] == b_adj - t_adj
        assert chip.shape[1] == r_adj - l_adj

        # Get the old image's face
        old_face = image[t:b, l:r, :]
        of_h, of_w, _ = old_face.shape
        assert of_h == adj_bbox_h, f"Old face chip h: {of_h}, new h: {adj_bbox_h}"
        assert of_w == adj_bbox_w, f"Old face chip w: {of_w}, new w: {adj_bbox_w}"

        # Chip out from the new image chip
        new_face = chip[bbox_new_t:bbox_new_b, bbox_new_l:bbox_new_r, :]
        assert np.all(new_face == old_face)

        # RESIZING Function: If the chip is huge, it wastes storage space.
        # This will resize to a manageable size.

        ch, cw = chip.shape[:2]

        if (cw > self.max_dim or ch > self.max_dim):

            # For error checking
            h_w_ratio = ch / cw
            hw_aspect = h_w_ratio > 1
            old_bbox_aspect = adj_bbox_h / adj_bbox_w

            rescale = self.max_dim / np.max((ch, cw))

            new_h = int(ch * rescale)
            new_w = int(cw * rescale)

            chip = cv2.resize(chip, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Error checking
            ch2, cw2, _ = chip.shape
            new_hw_ratio = ch2 / cw2
            new_hw_aspect = new_hw_ratio > 1
            assert new_hw_aspect == hw_aspect

            # Rescale the bounding box
            bbox_new = [int(f * rescale) for f in bbox_new[0]]
            bbox_new = [tuple(bbox_new)]

            new_bbox_h = np.abs(bbox_new[0][0] - bbox_new[0][2])
            new_bbox_w = np.abs(bbox_new[0][1] - bbox_new[0][3])
            new_bbox_aspect = new_bbox_h / new_bbox_w
            assert np.abs(new_bbox_aspect - old_bbox_aspect) < 0.01

        ch, cw = chip.shape[:2]
        assert ch <= self.max_dim
        assert cw <= self.max_dim
        
        return {'chip': chip,
                'bbox_trbl': bbox_new}

        # t_diff = np.abs(bbox_new[0][0] - np.abs(t_adj - b_adj) / 6 )
        # r_diff = bbox_new[0][1] - np.abs( (r_adj - l_adj - np.abs(r_adj - l_adj) / 6 )) 
        # b_diff = bbox_new[0][2] - np.abs( (b_adj - t_adj - np.abs(t_adj - b_adj) / 6 ))
        # l_diff = np.abs(bbox_new[0][3] - np.abs(r_adj - l_adj) / 6 )
            
        # if t_adj != 0 and b_adj != imh:
        #     assert t_diff < 1, f'{bbox_new[0][0]} - np.abs({t_adj} - {b_adj}) / 6) | {t}, {r}, {b}, {l}'
        #     assert b_diff < 1, f'{bbox_new[0][2]} - np.abs( ({b_adj} - {t_adj} - np.abs({t_adj} - {b_adj}) / 6 )) | {t}, {r}, {b}, {l}'
        # if l_adj != 0 and r_adj != imw:
        #     assert r_diff < 1, f'{bbox_new[0][1]} - np.abs( ({r_adj} - {l_adj} - np.abs({r_adj} - {l_adj}) / 6 )) | {t}, {r}, {b}, {l}'
        #     assert l_diff < 1, f'np.abs({bbox_new[0][3]} - np.abs({r_adj} - {l_adj}) / 6 ) | {t}, {r}, {b}, {l}'
        
        # Chip out the image.


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
        data['bbox_trbl'] = chip_data['bbox_trbl']

        data['face_id'] = face.id
        data['img_file'] = face.source_image_file.filename
        data['date_taken'] = face.source_image_file.dateTaken
        data['date_modified'] = face.source_image_file.dateModified

        if self.test:
            print("writing to ", out_file)
            
        with open(out_file, 'wb') as fh:
            pickle.dump(data, fh)

        os.chmod(out_file, 0o664)


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
