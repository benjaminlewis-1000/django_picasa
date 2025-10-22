#! /usr/bin/env python

from face_manager.models import Person, Face
from face_manager.face_extract_encode import FaceExtractor
from filepopulator.models import ImageFile
from django.core.management.base import BaseCommand
import editdistance
from time import sleep
import numpy as np
import insightface 
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import torch
import torchvision.ops.boxes as bops
import cv2
import common


class Command(BaseCommand):

    def __init__(self):
        # self.iou_function = bops.complete_box_iou 
        # self.iou_function = bops.generalized_box_iou 
        # self.iou_function = bops.box_iou 
    
        """
        RESET: 
        reencoded = False
        """

    def handle(self, *args, **options):

        extractor = FaceExtractor()

        # extractor.reset_all_images()
        # extractor.starting_reset()
        
        extractor.find_and_encode_faces()

        exit()
