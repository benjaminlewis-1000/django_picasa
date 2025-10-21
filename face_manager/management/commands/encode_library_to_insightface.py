#! /usr/bin/env python

from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from face_manager.face_extract_encode import extract_faces
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
        self.IOU_thresh = 0.3
        self.iou_function = bops.distance_box_iou 
        # self.iou_function = bops.complete_box_iou 
        # self.iou_function = bops.generalized_box_iou 
        # self.iou_function = bops.box_iou 
    
        """
        RESET: 
        reencoded = False
        """

    def handle(self, *args, **options):

        # all_objs = Face.objects.all()
        pp = Face.objects.get(id=371160)
        source = pp.source_image_file.filename
        # Get the faces that are in the image already

        img_numpy = common.open_img_oriented(source, as_numpy = True)

        # Get existing faces, if any, attached to the image.
        # Put the bounding boxes of these faces into a 2D tensor.
        existing_faces = Face.objects.filter(source_image_file=pp.source_image_file)
        for ff in existing_faces: 
            tl = (ff.box_left, ff.box_top)
            br = (ff.box_right, ff.box_bottom)
            cv2.rectangle(img_numpy, tl, br, (255, 0, 0), 2)
            print(f"{ff.box_left}, {ff.box_top}, {ff.box_right}, {ff.box_bottom}")

        n_existing = len(existing_faces)
    
        existing_boxes = torch.zeros(n_existing, 4)
        for ex_idx in range(n_existing):
            ex_box = existing_faces[ex_idx]
            ex_box = torch.tensor([[ex_box.box_left, ex_box.box_top, ex_box.box_right, ex_box.box_bottom]], dtype=torch.float)
            existing_boxes[ex_idx, :] = ex_box            
        print(existing_boxes)
        del ex_idx
        
        face_data = extract_faces(source)
        n_detect = len(face_data)
        for ff in face_data:

            tl = (int(ff['bbox'][0]), int(ff['bbox'][1]))
            br = (int(ff['bbox'][2]), int(ff['bbox'][3]))
            print(tl, br)
            cv2.rectangle(img_numpy, tl, br, (255, 255, 0), 2)
            print(ff['bbox'], ff.keys(), ff['gender'], ff['age'])
        detect_boxes = torch.zeros(n_detect, 4)

        bgr_image = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        print("TODO: Stop saving image out")
        cv2.imwrite("/code/output_image.jpg", bgr_image)

        for det_idx in range(n_detect):
            dt_box = face_data[det_idx]
            dt_box = torch.tensor([dt_box['bbox']])
            detect_boxes[det_idx, :] = dt_box
        del det_idx
        
        print(detect_boxes)

        iou = self.iou_function(existing_boxes, detect_boxes)
        print(iou)

        # Now we do some cases... 
        iou = iou.numpy()
        
        # Suppress low IOUs
        iou[iou < self.IOU_thresh] = 0
        max_ious = np.max(iou, axis=1) # Max IOU for each existing detection

        # Case 1 & 2
        if np.min(max_ious) >= self.IOU_thresh:
            # Candidate rows/columns are places where the IOU is greater
            # than a threshold. 
            candidate_rows, candidate_cols = np.where(iou >= self.IOU_thresh)
            set_candidate_rows = list(set(candidate_rows.tolist()))
            set_candidate_rows.sort()

            # Case 1: All match one-to-one for IOUs. 
            if set_candidate_rows == np.arange(len(set_candidate_rows)).tolist():
                print(f"One-to-one matches acquired")

                # Make sure to find indices where InsightFace found new faces.
                column_maxs = np.max(iou, axis=0)
                print(iou, column_maxs)

                # This is the set of indices where a new face was detected by
                # InsightFace and needs to be added. 
                new_face_idcs = np.where(column_maxs == 0)[0]
                print("TODO: Add new faces")

                # Match existing faces to new data. This gives us an array
                # where the position in the array corresponds to the existing
                # face's index (position in existing_faces) and the value of that
                # position in the array is the newly detected face's index 
                # (position in face_data). Then we can go through and update. 
                matching_face_idcs = np.argmax(iou, axis=1)
                for ex_idx, new_idx in enumerate(matching_face_idcs):
                    print(ex_idx, new_idx)
                    existing_data = existing_faces[ex_idx]
                    new_data = face_data[new_idx]

                    self.update_existing_face_to_insightface(existing_data, new_data)

                print(matching_face_idcs)

            else:
                raise NotImplementedError("Not one-to-one match")

        elif np.min(max_ious) < self.IOU_thresh:
            raise NotImplementedError("Not implemented")

    def update_existing_face_to_insightface(self, existing_face, new_data):
        
        if not type(existing_face) == Face:
            raise TypeError(f"Existing face must be of type face_manager.models.Face, is {type(existing_face)}")
        if not type(new_data) == insightface.app.common.Face:
            raise TypeError(f"New data must be of type insightface.app.common.Face, is {type(new_data)}")

        # Compute the IOU, ensure that it's greater 
        dt_box = torch.tensor([new_data['bbox']])
        ex_box = torch.tensor([[existing_face.box_left, existing_face.box_top, \
            existing_face.box_right, existing_face.box_bottom]], dtype=torch.float)
        check_iou = float(self.iou_function(dt_box, ex_box)[0][0])

        if check_iou < self.IOU_thresh:
            raise ValueError(f"IOU between the existing and new faces was less than the IOU " + \
                f"threshold of {self.IOU_thresh}. Please check whether this was in error.")
        print("check iou: ", check_iou)

        detected_gender = new_data['gender']
        detected_age = new_data['age']
        new_embedding = new_data['embedding']
        assert len(new_embedding) == 512

        print(dir(existing_face))
        print(existing_face.face_encoding_512[:10])
        print(existing_face.weight_1)
        print(new_embedding[:10])
        print(new_data['bbox'])

        new_left, new_top, new_right, new_bottom = new_data['bbox']
        new_left = int(new_left)
        new_top = int(new_top)
        new_right = int(new_right)
        new_bottom = int(new_bottom)
        print(existing_face.box_left, existing_face.box_top, existing_face.box_right, existing_face.box_bottom)

        existing_face.face_encoding_512 = new_embedding.tolist()
        existing_face.box_left = new_left
        existing_face.box_top = new_top
        existing_face.box_right = new_right
        existing_face.box_bottom = new_bottom
        existing_face.detected_age = detected_age
        existing_face.detected_gender = detected_gender
        existing_face.reencoded = True
        existing_face.save()
        print("Face ID is: ", existing_face.id)
