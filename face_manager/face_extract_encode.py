#! /usr/bin/env python

from face_manager.models import Person, Face
from django.conf import settings
from filepopulator.models import ImageFile
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import common
import cv2
import insightface 
import numpy as np
import os
import time
import torch
import torchvision.ops.boxes as bops
from pyramidal_detector import PyramidalDetector

class FaceExtractor(object):
    """docstring for FaceExtractor"""
    def __init__(self):
        super(FaceExtractor, self).__init__()

        face_analysis = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
        # self.app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
        face_analysis.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

        self.app = PyramidalDetector(detector = face_analysis)
        
        self.IOU_thresh = 0.3
        self.iou_function = bops.distance_box_iou 
        self.blank_face_person = Person.objects.get(person_name = settings.BLANK_FACE_NAME)

        # Based on code at: 
        # https://github.com/deepinsight/insightface/blob/2a78baec428354883e0cda39c54b555a5ed8358a/cpp-package/inspireface/cpp/inspireface/include/inspireface/data_type.h#L285
        self.gender_map = {1: 'M', 0: 'F'}

        # The amount we go out beyond the InsightFace detection to clip the thumbnail.
        self.thumbnail_extension_mult = 2 

    def reset_all_images(self):
        """
        Only run if necessary. This will set all ImageFiles to unprocessed.
        """
        print()
        for i in range(3):
            print("#" * 120)
        print("This function is resetting all encoding progress. If you don't want this, " + \
            "please CTRL-C in the next five seconds.")
        for i in range(3):
            print("#" * 120)
        print()
        time.sleep(5)

        ImageFile.objects.all().update(isProcessed=False)
        Face.objects.all().update(reencoded=False)

    def starting_reset(self):
        """
        Run when there are only a few images to reset. 
        """

        processed_imgs = ImageFile.objects.filter(isProcessed=True)
        if len(processed_imgs) > 1000:
            self.reset_all_images()
        else:
            for pp in processed_imgs:
                pp.isProcessed = False
                pp.save()

            processed_faces = Face.objects.filter(reencoded = True)
            for pf in processed_faces:
                pf.reencoded = False
                pf.save()

    def find_and_encode_faces(self):
        """
        Workhorse function that finds faces for all unprocessed
        ImageFile objects. If the ImageFile already has Face objects
        attached to it, then the function will endeavor to match those faces
        with detected faces. 
        """

        # Get unprocessed files.
        unprocessed_imgs = ImageFile.objects.filter(isProcessed=False) # .order_by('?') 
        print(len(unprocessed_imgs), unprocessed_imgs[0].id)

        for img_obj in unprocessed_imgs:

            source_file = img_obj.filename
            print(source_file)
            img_numpy = common.open_img_oriented(source_file, as_numpy = True)

            # Get existing faces, if any, attached to the image.
            # Put the bounding boxes of these faces into a 2D tensor
            # which will be used to calculate IOU.
            existing_faces = Face.objects.filter(source_image_file=img_obj)
            n_existing = len(existing_faces)
            existing_boxes = torch.zeros(n_existing, 4)
            
            for face_idx, face_obj in enumerate(existing_faces): 
                
                ex_box = torch.tensor([[face_obj.box_left, face_obj.box_top, face_obj.box_right, face_obj.box_bottom]], dtype=torch.float)
                existing_boxes[face_idx, :] = ex_box

            print(existing_boxes)
            
            # Use the self.app InsightFace module to detect and classify 
            # faces in the image. Populate those bounding boxes into a 
            # tensor to eventually compute IOU. 
            face_data = self.app.get(img_numpy)
            n_detect = len(face_data)
            detect_boxes = torch.zeros(n_detect, 4)

            for det_face_idx, det_face_obj in enumerate(face_data):

                # Round the detection boxes to int precision
                face_data[det_face_idx]['bbox'] = [int(det_face_obj['bbox'][0]), 
                                                   int(det_face_obj['bbox'][1]), 
                                                   int(det_face_obj['bbox'][2]), 
                                                   int(det_face_obj['bbox'][3])]

                dt_box = torch.tensor(det_face_obj['bbox']).unsqueeze(0)
                detect_boxes[det_face_idx, :] = dt_box

            print(detect_boxes)

            if n_existing == 0 and n_detect == 0:
                # There is nothing to do here.
                img_obj.isProcessed = True
                img_obj.save()
                continue

            if n_existing == 0 and n_detect > 0:
                # TODO: Add new faces
                print("TODO: Implement here with only new faces")
                for det_face_obj in face_data:
                    self.add_new_face(det_face_obj, img_obj, img_numpy)

                # img_obj.isProcessed = True
                # img_obj.save()
                continue

            
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

                    for new_face_idx in new_face_idcs:
                        new_data = face_data[new_face_idx]
                        self.add_new_face(new_data, image_obj, img_numpy)

                    print(matching_face_idcs)
                    
                    # img_obj.isProcessed = True
                    # img_obj.save()

                else:
                    raise NotImplementedError("Not one-to-one match")

            elif np.min(max_ious) < self.IOU_thresh:
                raise NotImplementedError("Not implemented")

        exit()

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

        detected_gender = self.gender_map[new_data['gender']]

        detected_age = new_data['age']
        new_embedding = new_data['embedding']
        assert len(new_embedding) == 512

        # print(dir(existing_face))
        # print(existing_face.face_encoding_512[:10])
        # print(existing_face.weight_1)
        # print(new_embedding[:10])
        # print(new_data['bbox'])

        new_left, new_top, new_right, new_bottom = new_data['bbox']
        new_left = int(new_left)
        new_top = int(new_top)
        new_right = int(new_right)
        new_bottom = int(new_bottom)
        # print(existing_face.box_left, existing_face.box_top, existing_face.box_right, existing_face.box_bottom)

        existing_face.face_encoding_512 = new_embedding.tolist()
        existing_face.box_left = new_left
        existing_face.box_top = new_top
        existing_face.box_right = new_right
        existing_face.box_bottom = new_bottom
        existing_face.detected_age = detected_age
        existing_face.detected_gender = detected_gender
        existing_face.reencoded = True
        existing_face.save()
        # print("Face ID is: ", existing_face.id)

    def add_new_face(self, insight_detected_face, img_obj, img_numpy):

        if type(insight_detected_face) != insightface.app.common.Face:
            raise TypeError("Only face dictionaries detected by InsightFace are valid in this function.")

        if type(img_obj) != ImageFile:
            raise TypeError("We need a valid ImageFile object to associate with these faces")

        if type(img_numpy) != np.ndarray:
            raise TypeError("We need a numpy array for img_numpy")

        bb_l, bb_t, bb_r, bb_b = insight_detected_face['bbox']
        detected_gender = self.gender_map[insight_detected_face['gender']]
        detected_age = insight_detected_face['age']
        print(detected_gender, detected_age)
        
        new_face = Face()
        
        new_face.face_encoding = None # Not using this right now.
        new_face.face_encoding_512 = insight_detected_face['embedding'].tolist()
        new_face.declared_name = self.blank_face_person
        new_face.written_to_photo_metadata = False
        new_face.reencoded = True
        new_face.box_top = bb_l
        new_face.box_bottom = bb_t
        new_face.box_left = bb_r
        new_face.box_right = bb_b
        new_face.source_image_file = img_obj
        new_face.dateTakenUTC = img_obj.dateTakenUTC
        new_face.detected_age = detected_age
        new_face.detected_gender = detected_gender

        face_thumbnail = self.get_square_face_img(insight_detected_face, img_obj, img_numpy)
        assert face_thumbnail is not None

        print(dir(insight_detected_face))
        # new_face.face_encoding_512 = insight_detected_face
        print(new_face)
        exit()

        new_face.save()
        settings.LOGGER.debug(f"New face id is: {new_face.id}")

    def get_square_face_img(self, insight_detected_face, img_obj, img_numpy):

        if type(insight_detected_face) != insightface.app.common.Face:
            raise TypeError("Only face dictionaries detected by InsightFace are valid in this function.")

        if type(img_obj) != ImageFile:
            raise TypeError("We need a valid ImageFile object to associate with these faces")

        if type(img_numpy) != np.ndarray:
            raise TypeError("We need a numpy array for img_numpy")

        bb_l, bb_t, bb_r, bb_b = insight_detected_face['bbox']
        print("BBOX", insight_detected_face['bbox'])

        img_h, img_w, _ = img_numpy.shape

        face_h = bb_b - bb_t
        face_w = bb_r - bb_l
        face_center_vert = (bb_b - bb_t) // 2 + bb_t
        face_center_horiz = (bb_r - bb_l) // 2 + bb_l
        assert face_h > 0
        assert face_w > 0
        assert face_center_vert > 0
        assert face_center_horiz > 0

        # Compute how much margin we have horizontally and vertically on the face.
        # This will be used to compute a thumbnail that doesn't extend beyond
        # the image. 
        vert_margin = np.min( (face_center_vert, img_h - face_center_vert))
        horiz_margin = np.min( (face_center_horiz, img_w - face_center_horiz))
        assert vert_margin > 0
        assert horiz_margin > 0

        detection_max_dim = np.max((face_h, face_w))
        max_allowable_margin = np.min((vert_margin, horiz_margin))
        ideal_thumbnail_margin = detection_max_dim * self.thumbnail_extension_mult // 2
        print("TODO: Need to figure out how to chip image, including check margin")
            
        # print(insight_detected_face)
# def extract_faces(filename: str) -> dict:
#     # Given a file name, use InsightFace to extract a dictionary of
#     # people and 512-dimensional vector encodings.

#     if not os.path.exists(filename):
#         raise FileNotFoundError(f"File {filename} not found")

#     img_numpy = common.open_img_oriented(filename, as_numpy=True)

    
#     s = time.time()
#     print(f"Starting encode at {s}")
#     faces = app.get(img_numpy)
#     print(f"Elapsed: {time.time() - s: .2f}")
    
#     return faces
