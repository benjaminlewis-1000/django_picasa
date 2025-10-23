#! /usr/bin/env python

from django.conf import settings
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from insightface.app import FaceAnalysis
from django.core.files.base import ContentFile
from insightface.data import get_image as ins_get_image
from io import BytesIO
from pyramidal_detector import PyramidalDetector
import common
import cv2
import hashlib
import insightface 
import numpy as np
import os
import time
import torch
import torchvision.ops.boxes as bops

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
        unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Pictures_In_Progress/2024/Family Texts/Resized_20221219_143329_20221220_015042.jpg')
        unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Completed/Pictures_finished/Nicholas\' Pictures/Christmas time and Goblin Valley/PC094087.JPG')
        unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Pictures_In_Progress/2023/Nathaniel Preschool/Bloomz_e4d23164-bcab-4615-ad57-a20dc169de1b.jpeg')
        unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Completed/Pictures_finished/Family Pictures/2016/February 2016/Meagan Leaves/_DSC0247.JPG')
        unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Pictures_In_Progress/Emily_amazon_uploads/2017-01-05_07-37-57_903.jpeg')
        unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Completed/Pictures_finished/Lewis Family Scans/Scan batch 2/1997_better/1997_00176A.jpg')
        unprocessed_imgs = ImageFile.objects.filter(isProcessed=False).order_by('?') 
        # unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Pictures_In_Progress/2024/Life/20240313_192916.jpg')

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
            
            # Use the self.app InsightFace module to detect and classify 
            # faces in the image. Populate those bounding boxes into a 
            # tensor to eventually compute IOU. 
            detected_faces = self.app.get(img_numpy)
            n_detect = len(detected_faces)
            detect_boxes = torch.zeros(n_detect, 4)

            for det_face_idx, det_face_obj in enumerate(detected_faces):

                # Round the detection boxes to int precision
                detected_faces[det_face_idx]['bbox'] = [int(det_face_obj['bbox'][0]), 
                                                   int(det_face_obj['bbox'][1]), 
                                                   int(det_face_obj['bbox'][2]), 
                                                   int(det_face_obj['bbox'][3])]

                dt_box = torch.tensor(det_face_obj['bbox']).unsqueeze(0)
                detect_boxes[det_face_idx, :] = dt_box

            if n_existing == 0 and n_detect == 0:
                # There is nothing to do here.
                img_obj.isProcessed = True
                img_obj.save()
                continue

            if n_existing == 0 and n_detect > 0:
                # TODO: Add new faces
                for det_face_obj in detected_faces:
                    self.add_new_face(det_face_obj, img_obj, img_numpy)

                img_obj.isProcessed = True
                img_obj.save()
                continue

            
            iou = self.iou_function(existing_boxes, detect_boxes)

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
                    # print(f"One-to-one matches acquired")

                    # Make sure to find indices where InsightFace found new faces.
                    column_maxs = np.max(iou, axis=0)
                    # print(iou, column_maxs)

                    # This is the set of indices where a new face was detected by
                    # InsightFace and needs to be added. 
                    new_face_idcs = np.where(column_maxs == 0)[0]

                    # Match existing faces to new data. This gives us an array
                    # where the position in the array corresponds to the existing
                    # face's index (position in existing_faces) and the value of that
                    # position in the array is the newly detected face's index 
                    # (position in detected_faces). Then we can go through and update. 
                    matching_face_idcs = np.argmax(iou, axis=1)
                    for ex_idx, new_idx in enumerate(matching_face_idcs):
                        # print(ex_idx, new_idx)
                        existing_data = existing_faces[ex_idx]
                        new_data = detected_faces[new_idx]

                        self.update_existing_face_to_insightface(existing_data, new_data)

                    for new_face_idx in new_face_idcs:
                        new_data = detected_faces[new_face_idx]
                        self.add_new_face(new_data, img_obj, img_numpy)

                    # print(matching_face_idcs)
                    
                    img_obj.isProcessed = True
                    img_obj.save()

                else:
                    raise NotImplementedError("Not one-to-one match")

            elif np.min(max_ious) < self.IOU_thresh:
                print("Existing faces: ", existing_boxes)
                print("Detected faces: ", detect_boxes)
                print("IOU: ", iou)
                raise NotImplementedError("Not implemented")

            # Assert that the image isProcessed flag is set
            assert img_obj.isProcessed == True, 'Image isProcessed flag was not set'
            # Get the number of faces associated with this object
            img_faces = Face.objects.filter(source_image_file = img_obj)
            assert len(img_faces) >= len(detected_faces)
            for face in img_faces:
                assert face.face_encoding_512 is not None
                assert face.reencoded == True

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

        img_h = existing_face.source_image_file.height
        img_w = existing_face.source_image_file.width

        if check_iou < self.IOU_thresh:
            raise ValueError(f"IOU between the existing and new faces was less than the IOU " + \
                f"threshold of {self.IOU_thresh}. Please check whether this was in error.")
        # print("check iou: ", check_iou)

        detected_gender = self.gender_map[new_data['gender']]

        detected_age = new_data['age']
        new_embedding = new_data['embedding']
        assert len(new_embedding) == 512

        new_left, new_top, new_right, new_bottom = new_data['bbox']
        new_left = np.max((0, int(new_left)))
        new_top = np.max((0, int(new_top)))
        new_right = np.min((int(new_right), img_w))
        new_bottom = np.min((int(new_bottom), img_h))
        assert new_left < new_right
        assert new_top < new_bottom

        existing_face.face_encoding_512 = new_embedding.tolist()
        existing_face.box_left = new_left
        existing_face.box_top = new_top
        existing_face.box_right = new_right
        existing_face.box_bottom = new_bottom
        existing_face.detected_age = detected_age
        existing_face.detected_gender = detected_gender
        existing_face.reencoded = True
        existing_face.save()

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
        # print(detected_gender, detected_age)
        
        new_face = Face()
        
        new_face.face_encoding = None # Not using this right now.
        new_face.face_encoding_512 = insight_detected_face['embedding'].tolist()
        new_face.declared_name = self.blank_face_person
        new_face.written_to_photo_metadata = False
        new_face.reencoded = True
        # new_face.box_top = bb_l
        # new_face.box_bottom = bb_t
        # new_face.box_left = bb_r
        # new_face.box_right = bb_b
        new_face.box_top = bb_t
        new_face.box_bottom = bb_b
        new_face.box_left = bb_l
        new_face.box_right = bb_r
        new_face.source_image_file = img_obj
        new_face.dateTakenUTC = img_obj.dateTakenUTC
        new_face.detected_age = detected_age
        new_face.detected_gender = detected_gender

        face_thumbnail = self.get_square_face_img(insight_detected_face, img_obj, img_numpy)
        assert face_thumbnail is not None
        assert type(face_thumbnail) is np.ndarray
        assert face_thumbnail.shape[0] == settings.FACE_THUMBNAIL_SIZE[0], f"Thumbnail size is {face_thumbnail.shape[:2]}, not {settings.FACE_THUMBNAIL_SIZE}"
        assert face_thumbnail.shape[1] == settings.FACE_THUMBNAIL_SIZE[1], f"Thumbnail size is {face_thumbnail.shape[:2]}, not {settings.FACE_THUMBNAIL_SIZE}"
        
        is_success, buffer_img = cv2.imencode(".jpg", face_thumbnail)
        # Save thumbnail to in-memory file as BytesIO
        temp_thumb = BytesIO(buffer_img)
        temp_thumb.seek(0)

        thumb_hash = self.hash_numpy_array(face_thumbnail)
        thumb_filename = f'{img_obj.pixel_hash}_{img_obj.file_hash}_facehash_{thumb_hash[:10]}.jpg'
        settings.LOGGER.debug(f"New face object is populated for file {img_obj.filename}, thumbnail hash {thumb_hash[:10]}, {thumb_filename}")
        # Load a ContentFile into the thumbnail field so it gets saved
        new_face.face_thumbnail.save(thumb_filename, ContentFile(temp_thumb.read())) 
        temp_thumb.close()

        new_face.save()
        settings.LOGGER.debug(f"New face id is: {new_face.id}")

    def get_square_face_img(self, insight_detected_face, img_obj, img_numpy):

        if type(insight_detected_face) != insightface.app.common.Face:
            raise TypeError("Only face dictionaries detected by InsightFace are valid in this function.")

        if type(img_obj) != ImageFile:
            raise TypeError("We need a valid ImageFile object to associate with these faces")

        if type(img_numpy) != np.ndarray:
            raise TypeError("We need a numpy array for img_numpy")

        img_h, img_w, _ = img_numpy.shape

        bb_l, bb_t, bb_r, bb_b = insight_detected_face['bbox']
        # print("BBOX", insight_detected_face['bbox'])

        bb_l = np.max((0, bb_l))
        bb_t = np.max((0, bb_t))
        bb_r = np.min((bb_r, img_w))
        bb_b = np.min((bb_b, img_h))

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
        actual_margin = np.min((ideal_thumbnail_margin, max_allowable_margin))
        # But we want the margin to be *at least* the bounding box max dimension.
        actual_margin = np.max((actual_margin, detection_max_dim // 2))

        chip_l = face_center_horiz - actual_margin
        chip_r = face_center_horiz + actual_margin
        chip_t = face_center_vert - actual_margin
        chip_b = face_center_vert + actual_margin

        chip_h = chip_b - chip_t
        chip_w = chip_r - chip_l

        left_pad = 0
        right_pad = 0
        top_pad = 0
        bot_pad = 0
        
        if chip_l < 0:
            left_pad = np.abs(chip_l)
            chip_l = 0
        if chip_r > img_w:
            right_pad = np.abs(chip_r - img_w)
            chip_r = img_w
        if chip_t < 0: 
            top_pad = np.abs(chip_t)
            chip_t = 0
        if chip_b > img_h:
            bot_pad = np.abs(chip_b - img_h)
            chip_b = img_h

        assert chip_l <= (bb_l + 1), f"chip_l {chip_l} !<= {bb_l + 1} | Chip TBLR: {chip_t}, {chip_b}, {chip_l}, {chip_r} || BB TBLR: {bb_t}, {bb_b}, {bb_l}, {bb_r} || IMH {img_h} IMW {img_w}"
        assert chip_t <= (bb_t + 1), f"chip_t {chip_t} !<= {bb_t + 1} | Chip TBLR: {chip_t}, {chip_b}, {chip_l}, {chip_r} || BB TBLR: {bb_t}, {bb_b}, {bb_l}, {bb_r} || IMH {img_h} IMW {img_w}"
        assert chip_r >= (bb_r - 1), f"chip_r {chip_r} !>= {bb_r - 1} | Chip TBLR: {chip_t}, {chip_b}, {chip_l}, {chip_r} || BB TBLR: {bb_t}, {bb_b}, {bb_l}, {bb_r} || IMH {img_h} IMW {img_w}"
        assert chip_b >= (bb_b - 1), f"chip_b {chip_b} !>= {bb_b - 1} | Chip TBLR: {chip_t}, {chip_b}, {chip_l}, {chip_r} || BB TBLR: {bb_t}, {bb_b}, {bb_l}, {bb_r} || IMH {img_h} IMW {img_w}"
        assert chip_l >= 0
        assert chip_t >= 0
        assert chip_r <= img_w
        assert chip_b <= img_h
        
        face_thumbnail = img_numpy[chip_t:chip_b, chip_l:chip_r]
        # Pad the image 
        face_thumbnail = np.pad(face_thumbnail, ((top_pad, bot_pad), (left_pad, right_pad), (0, 0)), 'constant')
        
        assert face_thumbnail.shape[0] == chip_h
        assert face_thumbnail.shape[1] == chip_w
        chip_h, chip_w, _ = face_thumbnail.shape
        assert chip_h == chip_w

        # Resize the image appropriately. 
        sq_thumb = cv2.cvtColor(face_thumbnail, cv2.COLOR_BGR2RGB)
        sq_thumb_resize = cv2.resize(sq_thumb, settings.FACE_THUMBNAIL_SIZE)
        assert sq_thumb_resize.shape[0] == settings.FACE_THUMBNAIL_SIZE[0]
        assert sq_thumb_resize.shape[1] == settings.FACE_THUMBNAIL_SIZE[1]

        return sq_thumb_resize

    def hash_numpy_array(self, arr):
        """
        Generates a hash for a NumPy array by combining its byte representation
        and shape.
        """
        # Convert the array's data to bytes
        arr_bytes = arr.reshape(-1).tobytes()
        # Get the array's shape as a tuple of integers
        arr_shape = arr.shape
        # Combine bytes and shape for hashing
        combined_data = arr_bytes + str(arr_shape).encode('utf-8')
        # Use a secure hash algorithm like SHA256
        return hashlib.sha256(combined_data).hexdigest()
            
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
