#! /usr/bin/env python

from django.conf import settings
from django.db.models import Q
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from insightface.app import FaceAnalysis
import django
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
        
        self.IOU_thresh = 0.3
        self.iou_function = bops.distance_box_iou 
        self.blank_face_person = Person.objects.get(person_name = settings.BLANK_FACE_NAME)

        self.app = PyramidalDetector(detector = face_analysis, iou_thresh = self.IOU_thresh)

        # Based on code at: 
        # https://github.com/deepinsight/insightface/blob/2a78baec428354883e0cda39c54b555a5ed8358a/cpp-package/inspireface/cpp/inspireface/include/inspireface/data_type.h#L285
        self.gender_map = {1: 'M', 0: 'F'}

        # The amount we go out beyond the InsightFace detection to clip the thumbnail.
        self.thumbnail_extension_mult = 2

        # Margin (as a fraction of the face's own box width/height, on each
        # side) used by reencode_missing_faces() when cropping around a
        # known box for a fresh single-pass detection. Matches the value
        # validated experimentally against real production faces.
        self.reencode_crop_margin_mult = 0.6

    @staticmethod
    def _flatten_kps(kps):
        """Flatten InsightFace's (5, 2) kps array to a plain 10-float list
        for storage in Face.kps, or None if no landmarks are available."""
        if kps is None:
            return None
        return np.asarray(kps, dtype=float).reshape(-1).tolist()

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

    def tiebreak_overlapping_bboxes(self, existing_bbox, det_bboxes, overlap_scores, considered_indices):

        if len(existing_bbox.shape) == 1:
            single_box = existing_bbox
            multiple_boxes = det_bboxes
        else:
            single_box = det_bboxes
            multiple_boxes = existing_bbox

        while multiple_boxes.shape[0] == 1:
            multiple_boxes = multiple_boxes.squeeze(0)

        assert len(single_box) == 4
        assert multiple_boxes.shape[1] == 4
        assert multiple_boxes.shape[0] > 1
        assert len(overlap_scores) == multiple_boxes.shape[0]
            
        assert type(single_box) == torch.Tensor
        assert type(multiple_boxes) == torch.Tensor
        assert type(overlap_scores) == np.ndarray
        assert len(considered_indices) == len(overlap_scores)
        assert type(considered_indices) == np.ndarray

        sl, st, sr, sb = single_box
        center_lr_single = np.abs(sr - sl) // 2 + sl
        center_tb_single = np.abs(sb - st) // 2 + st

        min_dist = 99999999999
        min_idx = -1
        for local_idx, mult_bbox in enumerate(multiple_boxes):
            # print(local_idx, mult_bbox)
            ml, mt, mr, mb = mult_bbox 
            center_lr_mult = np.abs(mr - ml) // 2 + ml
            center_tb_mult = np.abs(mb - mt) // 2 + mt

            distance = np.sqrt( (center_lr_single - center_lr_mult) ** 2 + (center_tb_single - center_tb_mult) ** 2 )
            if distance < min_dist:
                min_dist = distance
                min_idx = local_idx 
            # print(distance, min_dist, min_idx)
        # for rn in j

        closest_bbox_idx = considered_indices[min_idx]
        # print(closest_bbox_idx)
        return closest_bbox_idx


    def find_and_encode_faces(self):
        """
        Workhorse function that finds faces for all unprocessed
        ImageFile objects. If the ImageFile already has Face objects
        attached to it, then the function will endeavor to match those faces
        with detected faces. 
        """

        # # # unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Completed/Pictures_finished/Nicholas\' Pictures/Christmas time and Goblin Valley/PC094087.JPG') # SOLVED 
        # # # unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Pictures_In_Progress/2023/Nathaniel Preschool/Bloomz_e4d23164-bcab-4615-ad57-a20dc169de1b.jpeg') # SOLVED 
        # # # unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Completed/Pictures_finished/Family Pictures/2016/February 2016/Meagan Leaves/_DSC0247.JPG') # SOLVED 
        # # # unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Pictures_In_Progress/2024/Life/20240313_192916.jpg') # SOLVED 
        ### unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Pictures_In_Progress/Emily_amazon_uploads/2017-01-05_07-37-57_903.jpeg') # SOLVED except for getting encoding
        ### unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Pictures_In_Progress/2024/Family Texts/Resized_20221219_143329_20221220_015042.jpg')
        ### unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Completed/Pictures_finished/Lewis Family Scans/Scan batch 2/1997_better/1997_00176A.jpg')
        # # # unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Pictures_In_Progress/2024/Florida/20240415_125811.jpg') # Eigenvalues match is wrong ### Solved by reducing IOU threshold, it seems. 
        # unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Completed/Pictures_finished/2016/Yellowstone/DSC_9447.JPG') # SOLVED 
        # # # unprocessed_imgs = ImageFile.objects.filter(filename='/photos/Pictures_In_Progress/2021/Life/DSC_9825.JPG') # SOLVED - by setting the IOU threshold the same here and in pyramidal detector (0.3)
        # unprocessed_imgs = ImageFile.objects.filter(filename = '/photos/Pictures_In_Progress/preprocess/jessica_dropbox/2021-03-19 19.40.19.jpg') # No detected faces, some existing faces # SOLVED
        # Get unprocessed files.
        # unprocessed_imgs = ImageFile.objects.filter(filename = '/photos/Pictures_In_Progress/Family History/Funk and Cutler Scans by Ariel Benson/1965-03-19 Clarence Funk and Joan Henderson Wedding/1965-03-19 (16) Clarence Funk and Joan Henderson Wedding.jpg')
        # unprocessed_imgs = ImageFile.objects.filter(filename = '/photos/Pictures_In_Progress/2019/Ben Work Trips/London and Brighton/2019-05-17 19.45.53-1.jpg') # SOLVED: del moved inside for loop
        # unprocessed_imgs = ImageFile.objects.filter(filename = '/photos/Completed/Pictures_finished/Family Pictures/2011/2011 (12) December/Christmas 2011 (93).JPG') # SOLVED: del moved inside for loop
        # unprocessed_imgs = ImageFile.objects.filter(filename = '/photos/Pictures_In_Progress/2019/Ben Work Trips/London and Brighton/2019-05-13 18.58.02.jpg') # SOLVED: del moved inside for loop
        # unprocessed_imgs = ImageFile.objects.filter(filename = '/photos/Completed/Pictures_finished/Family Pictures/2012/7-2012/July Trip/Washington DC (93).JPG') # SOLVED: del moved inside for loop
        # unprocessed_imgs = ImageFile.objects.filter(filename = '/photos/aggregated/IMG_20260630_173524.jpg') # 
        # unprocessed_imgs = ImageFile.objects.filter(filename = '/photos/Phone Camera Uploads/20190324_201733.jpg') # 
        # unprocessed_imgs = ImageFile.objects.filter(filename = '/photos/Completed/Pictures_finished/Family Pictures/2013/Feb 2013/DSC_0405.JPG')
        unprocessed_imgs = ImageFile.objects.filter(isProcessed=False).order_by('?') 
        # unprocessed_imgs = ImageFile.objects.filter(filename = '/photos/Completed/Pictures_finished/Family Pictures/2008/2008 July/100_4251.JPG')
        # unprocessed_imgs = ImageFile.objects.filter(filename = '/photos/Completed/Pictures_finished/Misc Picture/Old Phone/20150123_201221.jpg')

        for img_obj in unprocessed_imgs:

            source_file = img_obj.filename
            print(source_file)
            try:
                img_numpy = common.open_img_oriented(source_file, as_numpy = True)
                assert len(img_numpy.shape) == 3
                # print(img_numpy.shape)

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
                # print(img_numpy.shape, type(img_numpy))
                detected_faces = self.app.get(img_numpy)
            except Exception as e:
                print("Error in processing!")
                settings.LOGGER.debug(f"Error processing image {source_file}: {str(e)}")
                print(f"Error processing image {source_file}: {str(e)}")
                # Mark isProcessed so this file isn't retried on every future
                # run (it never will decode differently), and record the
                # failure so it can be found/cleaned up later instead of
                # silently vanishing from the pipeline. Uses .update()
                # rather than img_obj.save() deliberately: ImageFile.save()
                # unconditionally re-decodes the image to recompute its
                # pixel hash (see filepopulator/models.py
                # _generate_md5_hash()), which would itself raise an
                # uncaught OSError on the same corrupted file (bug #6,
                # still open) right here in the handler for this failure.
                ImageFile.objects.filter(pk=img_obj.pk).update(
                    isProcessed=True,
                    image_load_failed=True,
                    image_load_error=str(e),
                )
                continue

            try:
                print(f"Found {len(detected_faces)} faces in image {source_file}")
                # print("Finished getting faces")
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

                # print(existing_boxes, "\n", detect_boxes)

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
                # print(iou, "|", existing_boxes, "|", detect_boxes)

                # Now we do some cases... 
                iou = iou.numpy()

                # Suppress low IOUs
                iou[iou < self.IOU_thresh] = 0
                # print(iou, len(iou), iou==[], type(iou), iou.shape)
                if iou.shape[1] == 0:
                    assert n_detect == 0
                    # print(iou, existing_boxes, detect_boxes)
                    # print(type(existing_faces), existing_faces)
                    # for jj in existing_faces:
                    #     print(type(jj))
                    self.update_list_of_no_matching_detects(existing_faces)
                    img_obj.save()

                else:
                    max_ious = np.max(iou, axis=1) # Max IOU for each existing detection
                    # print("max ious: ", max_ious)

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
                            # print(new_face_idcs)
    
                            # Match existing faces to new data. This gives us an array
                            # where the position in the array corresponds to the existing
                            # face's index (position in existing_faces) and the value of that
                            # position in the array is the newly detected face's index 
                            # (position in detected_faces). Then we can go through and update. 
                            matching_face_idcs = np.argmax(iou, axis=1)
                            for ex_idx, new_idx in enumerate(matching_face_idcs):
                                # print(ex_idx, new_idx)
                                # print(iou)
                                correlation_row = iou[ex_idx]
                                n_correlate = np.count_nonzero(correlation_row)
                                if n_correlate == 1:
                                    existing_data = existing_faces[ex_idx]
                                    new_data = detected_faces[new_idx]
                                elif n_correlate > 1:
                                    new_idcs = np.where(correlation_row > 0)[0]
                                
                                    selected_existing_bboxes = existing_boxes[ex_idx]
                                    selected_new_bboxes = detect_boxes[new_idcs]
                                    overlap_scores = correlation_row[new_idcs]
                                    # print(selected_existing_bboxes, selected_new_bboxes)
                                    # print(correlation_row)
                                    closest_idx = self.tiebreak_overlapping_bboxes(selected_existing_bboxes, selected_new_bboxes, overlap_scores, new_idcs)
                                    # print(closest_idx)
                                    # print("New idcs: ", new_idcs)
                                
                                    existing_data = existing_faces[ex_idx]
                                    new_data = detected_faces[closest_idx]
                                else:
                                    raise NotImplementedError("Should have at least one correlation")
                                # assert np.count_nonzero(iou[ex_idx]) == 1, \
                                #     f'An IOU match between detected and existing faces should only ' +\
                                #     'have one answer. This row was {iou[ex_idx]}'
    
                                self.update_existing_face_to_insightface(existing_data, new_data)
    
                            for new_face_idx in new_face_idcs:
                                new_data = detected_faces[new_face_idx]
                                self.add_new_face(new_data, img_obj, img_numpy)
    
                            # print(matching_face_idcs)
                        
                            img_obj.save()
    
                        else:
                            raise NotImplementedError("Not one-to-one match")
    
                    elif np.min(max_ious) < self.IOU_thresh:
                        row_sums = np.sum(iou > self.IOU_thresh, axis=1)
                        nonzero_rows = np.where(row_sums)[0]
                        # print("NZ rows", nonzero_rows)
    
                        # Handle matching rows
                        for rn in nonzero_rows:
                            row = iou[rn]
                            # print("RN = ", rn, row)
                            # assert np.count_nonzero(row) == 1
                            existing_idx = int(rn)
                            if np.count_nonzero(row) > 1:
                                # print("TODO")
                                existing_bbox = existing_boxes[existing_idx, :]
                                # print(existing_bbox)
                                nz_cols = np.where(row > 0)[0]
                                # print(nz_cols)
                                detect_bboxes = detect_boxes[nz_cols, :]
                                # print(detect_bboxes)
                                nz_scores = row[nz_cols]
                                # print(nz_scores)
                                detected_idx = self.tiebreak_overlapping_bboxes(existing_bbox, detect_bboxes, nz_scores, nz_cols)
                                # print(detected_idx)
                                assert row[detected_idx] > 0
                            elif np.count_nonzero(row) == 1:
                                detected_idx = np.argmax(row)
                            else:
                                assert np.count_nonzero(row) == 0
                                raise ValueError('No overlapping detected and existing boxes - you shouldn\'t get here')

                            existing_data = existing_faces[existing_idx]
                            new_data = detected_faces[detected_idx]
                            self.update_existing_face_to_insightface(existing_data, new_data)
                            del existing_idx, row, detected_idx

                        # Handle any new detections from InsightFace that were not previously there.
                        # print(iou)
                        column_maxs = np.max(iou, axis=0)
                        # print(column_maxs)
                        new_face_idcs = np.where(column_maxs == 0)[0]
                        # print(new_face_idcs)
    
                        for new_face_idx in new_face_idcs:
                            new_data = detected_faces[new_face_idx]
                            self.add_new_face(new_data, img_obj, img_numpy)
            
                        zero_row_sums = np.sum(iou, axis=1)
                        zero_rows = np.where(zero_row_sums == 0)[0]
                        # print("Zero rows", zero_rows, zero_row_sums)
                        img_h, img_w, _ = img_numpy.shape 

                        no_match_list = [existing_faces[int(idx)] for idx in zero_rows]
                        no_match_bbox = existing_boxes[zero_rows]
                        # print(no_match_bbox, existing_boxes)
                        check_iou = self.iou_function(no_match_bbox, detect_boxes)
                        assert torch.all(check_iou < self.IOU_thresh)
                        self.update_list_of_no_matching_detects(no_match_list)
    
                        # print("TODO: Handle zero-cols and zero-rows")
                        # print("max_IOUs", max_ious)
                        # print("Existing faces: ", existing_boxes)
                        # print("Detected faces: ", detect_boxes)
                        # pr    int("IOU: ", iou)
                        img_obj.save()
                        # raise NotImplementedError("Not implemented")
    
                # Get the number of faces associated with this object
                img_faces = Face.objects.filter(source_image_file = img_obj)
                # print(len(img_faces), len(detected_faces))
                assert len(img_faces) >= len(detected_faces), f"{len(img_faces)} is not >= {len(detected_faces)}"

                img_obj.isProcessed = True
                img_obj.save()
                # Assert that the image isProcessed flag is set
                assert img_obj.isProcessed == True, 'Image isProcessed flag was not set'

                for face in img_faces:
                    assert face.face_encoding_512 is not None
                    assert len(face.face_encoding_512) == 512
                    assert face.reencoded == True
            except Exception as e:
                # A failure anywhere in the IOU-matching logic below
                # (unhandled combinations the original author left as
                # NotImplementedError/ValueError/bare asserts) must not
                # abort the entire scheduled run -- every other
                # already-queued image would otherwise silently never
                # get attempted either, via face_manager.tasks.
                # process_faces()'s outer bare except. Leave isProcessed
                # unset (not marked failed) so this image is retried on
                # the next run rather than silently dropped -- these are
                # valid images that should eventually process once the
                # underlying matching logic is fixed.
                settings.LOGGER.error(f"Error matching faces for image {source_file}: {str(e)}")
                print(f"Error matching faces for image {source_file}: {str(e)}")
                continue

        # print("DONE")



    def update_list_of_no_matching_detects(self, unmatched_existing_faces: list):
        if type(unmatched_existing_faces) not in [django.db.models.query.QuerySet, list]:
            raise TypeError("Input must be a list or QuerySet")
        if len(unmatched_existing_faces) <= 0:
            raise ValueError("Input must be a non-zero list")
    
        img_h = unmatched_existing_faces[0].source_image_file.height
        img_w = unmatched_existing_faces[0].source_image_file.width
    
        for face_obj in unmatched_existing_faces:
            if type(face_obj) is not Face:
                raise TypeError("List must be objects of type Face, from face_manager.model")

            orig_left, orig_top = face_obj.box_left, face_obj.box_top
            orig_right, orig_bottom = face_obj.box_right, face_obj.box_bottom
            orientation = face_obj.source_image_file.orientation

            # 90/270-degree rotations (orientation 6/8) swap width and
            # height. A pre-existing decode-path disagreement (fixed by
            # this session's EXIF-orientation consolidation --
            # face_extract_encode's old common.open_img_oriented() call
            # could disagree with filepopulator's _init_image() about a
            # rotated image's true width/height) could compute a face's box
            # against the *unrotated* (swapped) dimensions instead of the
            # image's actual current ones. Bounds-checking alone can't
            # catch every case of this -- a swapped-frame box can still
            # happen to fit the current dimensions by coincidence -- so two
            # signals are used together:
            #  1. The box doesn't fit the current dims at all but does fit
            #     the swapped dims -- an unambiguous sign it was computed
            #     in the wrong coordinate frame.
            #  2. On an orientation 6/8 image, this face has never been
            #     successfully re-matched to a real detection since it was
            #     created (reencoded=False) -- its box has never actually
            #     been verified under the corrected decode path, even if it
            #     happens to still fit the current dimensions.
            # Signal 2 only applies to orientation 6/8, since no
            # width/height swap is possible for orientation 1/3 -- a stale
            # box there is instead caught by the degenerate-after-clamp
            # check below.
            fits_current = (0 <= orig_left < orig_right <= img_w) and (0 <= orig_top < orig_bottom <= img_h)
            fits_swapped = (0 <= orig_left < orig_right <= img_h) and (0 <= orig_top < orig_bottom <= img_w)
            untrusted_orientation = orientation in (6, 8) and not face_obj.reencoded

            if (not fits_current and fits_swapped) or untrusted_orientation:
                settings.LOGGER.error(
                    f"Deleting Face {face_obj.id} on {face_obj.source_image_file.filename}: "
                    f"its stored box ({orig_left}, {orig_top}, {orig_right}, {orig_bottom}) is "
                    f"untrustworthy against this orientation-{orientation} image's current "
                    f"dimensions ({img_w}x{img_h}) -- likely computed in a different coordinate "
                    f"space by a decode path that has since been fixed."
                )
                print(f"Deleting geometrically-untrustworthy stale Face {face_obj.id}")
                face_obj.delete()
                continue

            face_obj.face_encoding_512 = settings.NON_DETECTED_FACE_ENCODING
            face_obj.box_left = np.max((0, int(orig_left)))
            face_obj.box_top = np.max((0, int(orig_top)))
            face_obj.box_right = np.min((int(orig_right), img_w))
            face_obj.box_bottom = np.min((int(orig_bottom), img_h))

            # Regression fix: even after the checks above, clamping each
            # coordinate independently (rather than rejecting the box
            # outright) can still produce a degenerate box in edge cases
            # not caught by either signal (e.g. orientation 1/3, where no
            # width/height swap is possible but a box can still be stale
            # for other reasons). Keep this as a final safety net so
            # Face.save()'s ValidationError can never again crash and abort
            # the entire scheduled run the way it used to via
            # face_manager.tasks.process_faces()'s bare `except:`.
            if face_obj.box_right <= face_obj.box_left or face_obj.box_bottom <= face_obj.box_top:
                settings.LOGGER.error(
                    f"Deleting Face {face_obj.id} on {face_obj.source_image_file.filename}: "
                    f"its stored box ({face_obj.box_left}, {face_obj.box_top}, "
                    f"{face_obj.box_right}, {face_obj.box_bottom}) doesn't fit the image's "
                    f"current dimensions ({img_w}x{img_h}) even after clamping."
                )
                print(f"Deleting geometrically-invalid stale Face {face_obj.id}")
                face_obj.delete()
                continue

            face_obj.reencoded = True
            face_obj.save()

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
        existing_face.kps = self._flatten_kps(new_data['kps'])
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

        new_face.face_encoding_512 = insight_detected_face['embedding'].tolist()
        new_face.kps = self._flatten_kps(insight_detected_face['kps'])
        new_face.declared_name = self.blank_face_person
        new_face.written_to_photo_metadata = False
        new_face.reencoded = True

        img_h, img_w, _ = img_numpy.shape 
        
        bb_l = np.max((0, int(bb_l)))
        bb_t = np.max((0, int(bb_t)))
        bb_r = np.min((int(bb_r), img_w))
        bb_b = np.min((int(bb_b), img_h))           

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

    def reencode_missing_faces(self):
        """
        Re-encode faces that have no *usable* face_encoding_512 but are no
        longer declared .ignore/.realignore -- e.g. a confirmed-ignore face
        whose embedding was cleared to save storage, then reassigned away
        from that sentinel because it turned out to be a real person after
        all.

        "No usable embedding" means either NULL, or the
        settings.NON_DETECTED_FACE_ENCODING sentinel ([-999]*512) that
        update_list_of_no_matching_detects() stamps onto a face whose
        existing box wasn't matched to any detection during a full-image
        reprocessing pass -- a real, declared-to-a-real-person face with
        this sentinel is not "ignored," just stuck with a placeholder that
        looks like real data to anything that doesn't check for it
        specifically (found 2026-09-03: 1,207 faces database-wide carried
        this sentinel, only 1 of which the existing
        cleanup_chronically_unmatched_faces command's orientation-6/8-only
        scope happened to also catch -- the other 1,206 were invisible to
        every existing cleanup path).

        Deliberately filters on face_encoding_512 lacking real data this
        way, not on the `reencoded` flag -- that flag tracks whether a
        face's *box* came from the modern insightface pipeline, which is
        orthogonal to whether it currently has a usable embedding.

        Two paths, in order of preference:
          1. If the face has stored `kps` (5 landmark points, in the source
             image's own pixel coordinate space -- see Face.kps), replay
             the exact original alignment directly through the recognition
             model. No detection needed, and the result is the same
             embedding the original detection produced (verified against
             real production faces: cosine similarity ~1.0 to a same-run
             reference), since the only source of error in the crop-based
             path below is landmark re-detection noise, which this path
             skips entirely.
          2. Otherwise (an older face, detected before Face.kps existed),
             crop tightly around the known bounding box and run a single
             detection+recognition pass on just that crop. This is the
             same approach validated experimentally against real
             production faces (median ~0.97 cosine similarity to the
             original embedding for the actual .ignore/.realignore
             population, but a real tail -- roughly 1 in 6 such faces find
             no detection at all, and a few percent land far off).
             Deliberately does NOT fall back to full-image detection +
             IOU-matching (the approach find_and_encode_faces() uses) --
             for a face already this hard to redetect in isolation, that's
             unlikely to do meaningfully better and costs far more (a
             full-image pass takes seconds; a small crop takes well under
             one).

        A face that still finds no detection in its crop gets
        settings.REENCODE_DEFAULT_ENCODING (a neutral, valid, unit-norm
        vector) rather than being left NULL -- NULL is exactly what this
        method selects on, so leaving it NULL would retry it forever on
        every future run.
        """
        faces_to_reencode = Face.objects.filter(
            Q(face_encoding_512__isnull=True) | Q(face_encoding_512=settings.NON_DETECTED_FACE_ENCODING)
        ).exclude(
            declared_name__person_name__in=[settings.SOFT_IGNORE_NAME, '.realignore']
        ).select_related('source_image_file')

        rec_model = self.app.app.models['recognition']

        for face in faces_to_reencode:
            img_obj = face.source_image_file
            if img_obj is None or not img_obj.filename or not os.path.exists(img_obj.filename):
                settings.LOGGER.error(f"Cannot re-encode Face {face.id}: source image missing or unset.")
                continue

            try:
                img = common.open_img_oriented(img_obj.filename, as_numpy=True)
            except Exception as e:
                settings.LOGGER.error(f"Cannot re-encode Face {face.id}: failed to open image ({e}).")
                continue
            if img is None or len(img.shape) != 3:
                settings.LOGGER.error(f"Cannot re-encode Face {face.id}: image failed to decode.")
                continue

            if face.kps is not None:
                kps_arr = np.array(face.kps, dtype=np.float32).reshape(5, 2)
                insight_face = insightface.app.common.Face(kps=kps_arr)
                embedding = rec_model.get(img, insight_face)
                face.face_encoding_512 = embedding.tolist()
                face.reencoded = True
                face.save()
                continue

            h, w, _ = img.shape
            bl, bt, br, bb = face.box_left, face.box_top, face.box_right, face.box_bottom
            if bl is None or br is None or br <= bl or bb <= bt:
                settings.LOGGER.error(f"Cannot re-encode Face {face.id}: invalid stored box.")
                continue

            box_w, box_h = br - bl, bb - bt
            mx = int(box_w * self.reencode_crop_margin_mult)
            my = int(box_h * self.reencode_crop_margin_mult)
            cl, ct = max(0, bl - mx), max(0, bt - my)
            cr, cb = min(w, br + mx), min(h, bb + my)
            crop = img[ct:cb, cl:cr]

            if crop.shape[0] < 20 or crop.shape[1] < 20:
                face.face_encoding_512 = settings.REENCODE_DEFAULT_ENCODING
                face.save()
                continue

            dets = self.app.app.get(crop)
            if len(dets) == 0:
                face.face_encoding_512 = settings.REENCODE_DEFAULT_ENCODING
                face.save()
                continue

            # Several faces may be found in the crop (e.g. a neighboring
            # face partly included by the margin) -- the one closest to
            # the crop's center is our known face, since we cropped
            # symmetrically around its own box.
            crop_cx, crop_cy = crop.shape[1] / 2, crop.shape[0] / 2
            best, best_dist = None, None
            for d in dets:
                dl, dt, dr, db = d['bbox']
                dcx, dcy = (dl + dr) / 2, (dt + db) / 2
                dist = (dcx - crop_cx) ** 2 + (dcy - crop_cy) ** 2
                if best_dist is None or dist < best_dist:
                    best_dist, best = dist, d

            face.face_encoding_512 = best['embedding'].tolist()
            # Translate the freshly-detected kps from crop-local back to
            # the source image's absolute coordinates (matching how
            # add_new_face()/update_existing_face_to_insightface() store
            # them), so a FUTURE re-encode of this same face can use the
            # exact-replay path above instead of cropping again.
            if best['kps'] is not None:
                abs_kps = np.array(best['kps'], dtype=float) + np.array([cl, ct])
                face.kps = self._flatten_kps(abs_kps)
            face.reencoded = True
            face.save()

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
