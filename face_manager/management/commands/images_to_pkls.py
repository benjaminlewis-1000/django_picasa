#! /usr/bin/env python

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count
from django.db.models import Q
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from itertools import chain
from PIL import Image  
from random import randint
import cv2
import numpy as np
import os
import pickle
import shutil
import time

def get_and_resize(in_path, out_path):
    im = Image.open(in_path)
    newsize = (224, 224)
    im = im.resize(newsize, resample=Image.LANCZOS)
    im.save(out_path)


class Command(BaseCommand):

    def __init__(self):
        super(Command).__init__()

        self.min_faces = 50
        self.out_dir = '/code/MISC_DATA/image_pkls'
        self.alt_out_dir = '/photos/pkls'

    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

    def img_to_pkl(self, face, set_ign_idx = False):

        if not set_ign_idx:
            out_file = os.path.join(self.out_dir, f'imchip_{face.id}.pkl')
            out_alt_file = os.path.join(self.alt_out_dir, f'imchip_{face.id}.pkl')
        else:
            out_file = os.path.join(self.out_dir, f'imchip_ign_{face.id}.pkl')
            out_alt_file = os.path.join(self.alt_out_dir, 'ignore', f'imchip_ign_{face.id}.pkl')

        if os.path.exists(out_file) or os.path.exists(out_alt_file):
            # print(f"File {out_file} already good")
            return

        data = {}
        data['person_name'] = face.declared_name.person_name
        if not set_ign_idx:
            data['index'] = self.names_list.index(face.declared_name)
        else:
            data['index'] = -999
        data['left'] = face.box_left
        data['right'] = face.box_right
        data['bottom'] = face.box_bottom
        data['top'] = face.box_top
        data['width'] = face.box_right - face.box_left
        data['height'] = face.box_bottom - face.box_top
        data['face_id'] = face.id
        data['img_file'] = face.source_image_file.filename
        data['date_taken'] = face.source_image_file.dateTaken
        data['date_modified'] = face.source_image_file.dateModified

        # Read in the image
        image = cv2.imread(data['img_file'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_height, im_width = image.shape[:2]

        # Calculate how much bigger it needs to be for rotating - sqrt(2) is ideal. 
        max_extent = max(data['width'], data['height'])
        scale_up_size = int(np.ceil(np.sqrt(2) * max_extent))

        width_add_nominal = (scale_up_size - data['width']) // 2
        margin_left = data['left']
        margin_right = im_width - data['right']
        width_add_final = np.min((margin_left, margin_right, width_add_nominal))

        chip_left = data['left'] - width_add_final
        chip_right = data['right'] + width_add_final

        height_add_nominal = (scale_up_size - data['height']) // 2
        margin_top = data['top']
        margin_bottom = im_height - data['bottom']
        height_add_final = np.min((height_add_nominal, margin_top, margin_bottom))

        chip_top = data['top'] - height_add_final
        chip_bottom = data['bottom'] + height_add_final

        img_chipped = image[chip_top:chip_bottom, chip_left:chip_right]

        h, w = img_chipped.shape[:2]
        if h == 0 or w == 0:
            return
        # You can get back the image chip by just getting the center point and 
        # taking the width//2 and height//2 from that. 

        if scale_up_size > 800:
            img_chipped = self.image_resize(img_chipped, height=800)

        print(img_chipped.shape)
        data['chipped_image'] = img_chipped

        with open(out_file, 'wb') as fh:
            pickle.dump(data, fh)


    def handle(self, *args, **options):

        names = Person.objects.annotate(c=Count('face_declared')) \
            .filter(c__gt=self.min_faces) \
            .filter(~Q(person_name__in=settings.IGNORED_NAMES) ) 

        self.names_list = list(names)

        # images = Face.objects.filter(declared_name__in=names).order_by('?')

        # cnt = 0
        # n_imgs = images.count()
        # for p_img in images.iterator():
        #     if cnt % 500 == 0 and cnt > 0:
        #         print(f"{cnt}/{n_imgs} | {cnt/n_imgs * 100:.2f}%")
        #     cnt += 1

        #     self.img_to_pkl(p_img)

        # exit()

        criterion_rejected = Q( declared_name__person_name__in=['.ignore', '.realignore'])
        ign_faces = Face.objects.filter(criterion_rejected).order_by('?')

        cnt = 0
        n_imgs = ign_faces.count()
        for ign in ign_faces.iterator():
            if cnt % 500 == 0 and cnt > 0:
                print(f"{cnt}/{n_imgs} | {cnt/n_imgs * 100:.2f}%")
            cnt += 1
            self.img_to_pkl(ign, True)

        # ignore_dir = '/code/MISC_DATA/ignore_chips'
        # try:
        #     os.makedirs(ignore_dir)
        # except:
        #     pass