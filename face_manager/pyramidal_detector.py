#! /usr/bin/env python

from insightface.app import FaceAnalysis
import insightface 
import numpy as np
import torch
import torchvision.ops.boxes as bops

class PyramidalDetector():
    """
    A class that breaks an image into n x n rectangles, including the original
    image as a 1x1 square, and runs a FaceAnalysis detector on each level.
    It then de-duplicates detections across levels and rectangles to find an 
    'optimal' detection for each face. 
    """
    def __init__(self, detector=None):
        super(PyramidalDetector, self).__init__()

        if detector is None:
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
            self.app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU
        else:
            if type(detector) != FaceAnalysis:
                raise TypeError(f"Provided detector argument needs to be a insightface.app.FaceAnalysis, but was given {type(detector)}")
            self.app = detector

        # Percent of the image to overlap in each direction. 
        self.pct_overlap = 0.06
        self.iou_function = bops.distance_box_iou
        self.iou_thresh = 0.5

        # Set of pyramidal levels. E.g. 1 in the list is a 1x1 grid, 2 is a 2x2 grid, 3 is a 3x3 grid, etc.
        # We will run detections of all grids listed in this list, then
        # combine the results together with linear algebra. 
        self.cut_list = [1, 3]

    def get(self, np_image: np.ndarray) -> list:
        if type(np_image) != np.ndarray:
            raise TypeError('Import image must be a numpy array.')
        if not bool(np.all(np.array(self.cut_list) >= 1)):
            raise ValueError("All values in the self.cut_list parameter must be >= 1.")
        if not np.array(self.cut_list).dtype == int:
            raise ValueError("All values in the self.cut_list parameter must be integers.")
        if 1 not in self.cut_list:
            raise ValueError("Need to have the value of 1 in self.cut_list")

        height, width, _ = np_image.shape

        overlapping_detections = []

        for cut_dims in self.cut_list:
            # Figure out the start and end row and columns for this. 

            if cut_dims == 1:
                # Just run the image
                faces_at_level = self.app.get(np_image)
                chip_h, chip_w, _ = np_image.shape
                for idx in range(len(faces_at_level)):
                    faces_at_level[idx]['detect_pyr_level'] = cut_dims
                    faces_at_level[idx]['bbox'][0] = int(faces_at_level[idx]['bbox'][0]) # Left
                    faces_at_level[idx]['bbox'][2] = int(faces_at_level[idx]['bbox'][2]) # Right
                    faces_at_level[idx]['bbox'][1] = int(faces_at_level[idx]['bbox'][1]) # Top
                    faces_at_level[idx]['bbox'][3] = int(faces_at_level[idx]['bbox'][3]) # Bottom
                    # Even though it may technically go off-screen, this chip
                    # is always considered "on-screen" since there was no additional
                    # information to be had. 
                    faces_at_level[idx]['off_screen'] = True
                    
                overlapping_detections.extend(faces_at_level)

            else:
                subbox_width = width / cut_dims
                subbox_height = height / cut_dims
                width_expansion = subbox_width * self.pct_overlap
                height_expansion = subbox_height * self.pct_overlap

                for x_offset in range(cut_dims):
                    for y_offset in range(cut_dims):

                        # Calculate the edges of the sub-box.
                        left_edge = x_offset * subbox_width
                        right_edge = (x_offset + 1) * subbox_width
                        # Get expanded values
                        left_edge = int(max(0, left_edge - width_expansion))
                        right_edge = int(min(width, right_edge + width_expansion))

                        top_edge = y_offset * subbox_height
                        bottom_edge = (y_offset + 1) * subbox_height
                        top_edge = int(max(0, top_edge - height_expansion))
                        bottom_edge = int(min(height, bottom_edge + height_expansion))

                        assert left_edge < right_edge
                        assert top_edge < bottom_edge

                        assert (bottom_edge - top_edge) <= int(subbox_height * (1 + self.pct_overlap * 2) + 1)
                        assert (right_edge - left_edge) <= int(subbox_width * (1 + self.pct_overlap * 2) + 1)

                        # Cut out the chip. 
                        chip_part = np_image[top_edge:bottom_edge, left_edge:right_edge]
                        chip_h, chip_w, _ = chip_part.shape

                        # Detect on sub-image. Indicate whether the detection went off
                        # the edge of the chip, and adjust the bounding box to its
                        # proper location in the larger image. 
                        faces_at_level = self.app.get(chip_part)
                        for idx in range(len(faces_at_level)):
                            faces_at_level[idx]['detect_pyr_level'] = cut_dims
                            bbox_l, bbox_t, bbox_r, bbox_b = faces_at_level[idx]['bbox']
                            if bbox_l < 0 or bbox_t < 0 or bbox_r > chip_w or bbox_b > chip_h:
                                faces_at_level[idx]['off_screen'] = True
                            else:
                                faces_at_level[idx]['off_screen'] = False
                            faces_at_level[idx]['bbox'][0] = int(bbox_l + left_edge) # Left
                            faces_at_level[idx]['bbox'][2] = int(bbox_r + left_edge) # Right
                            faces_at_level[idx]['bbox'][1] = int(bbox_t + top_edge) # Top
                            faces_at_level[idx]['bbox'][3] = int(bbox_b + top_edge) # Bottom

                        overlapping_detections.extend(faces_at_level)

        bboxes = [det['bbox'] for det in overlapping_detections]
        bboxes = torch.tensor(np.array(bboxes))

        if len(overlapping_detections) == 0:
            return overlapping_detections

        iou = self.iou_function(bboxes, bboxes)
        binary_iou = torch.gt(iou, self.iou_thresh).to(torch.float)
        # Binary IOU matrix should be symmetric, so assert that it is
        torch.all(binary_iou - binary_iou.T == 0)
        # Compute the eigenvalues of the binary matrix. The number of non-
        # zero (or non-numerical noise) eigenvalues is equal to the rank 
        # of the matrix, and since there are identical rows/columns in a
        # matrix with overlapping IOUs, the rank is equal to the number of people.
        rank = torch.linalg.matrix_rank(binary_iou)

        # Find the unique rows. Now that we've binarized the IOU 
        # matrix, any faces that overlap and are greater than the 
        # threshold have a binary IOU of 1, otherwise 0. This gives
        # our matrix symmetry, and it also means that faces that 
        # overlap will have the exact same rows/columns for the index
        # corresponding to that face. We can then find unique rows
        # (or columns, but rows are easier to work with in some respects)
        # which then are going to be groups of correlated detections.
        unique_rows = torch.unique(binary_iou, dim=0)

        assert len(unique_rows) == rank, 'Rank and number of unique rows has a discrepancy.'
        sums = torch.sum(unique_rows, dim=1).to(int)
        # Get the eigenvalues of the matrix, which we are going to use
        # to check our assumptions. 
        eig = torch.linalg.eigh(binary_iou)
        eigenvalues = eig.eigenvalues
        # Sanity check - values of large eigenvalues
        # should be equal in quantity, if not order,
        # as the sums of unique rows. 
        large_eigens = eigenvalues[torch.where(eigenvalues > 0.01)]
        large_eigens = torch.round(large_eigens)
        large_eigens = [int(ev) for ev in large_eigens]
        assert set(large_eigens) - set(sums.tolist()) == set(), 'Assumption that eigenvalues and rows sums match is inaccurate.'

        # Now we have sets of unique rows. On a face-by-face basis, let's decide on the "best"
        # face and encoding, then return those.
        deduplicated_faces = []
        for face_group_row in unique_rows:
            group_idcs = torch.where(face_group_row)[0]
            face_group = [overlapping_detections[f] for f in group_idcs]
            if len(face_group) == 1:
                deduplicated_faces.append(face_group[0])
            else:
                det_levels = np.array([f['detect_pyr_level'] for f in face_group])
                on_screen = np.array([not f['off_screen'] for f in face_group])
                off_screen = np.array([f['off_screen'] for f in face_group])
                bboxes = [f['bbox'] for f in face_group]

                bbox_size = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes])
                highest_detection = det_levels == np.max(det_levels)
                zoomed_detection = det_levels > 1

                # Sanity check - all bounding boxes we are looking at satisfy our
                # IOU parameters
                bbox_tensor = torch.tensor(np.array(bboxes))
                iou_check = self.iou_function(bbox_tensor, bbox_tensor)
                assert torch.all(iou_check > self.iou_thresh)

                if len(self.cut_list) > 2:
                    raise NotImplementedError("There should be more logic here for intermediate zoom levels, but I haven't implemented.")

                # If it was off-screen, adjust its detection level to be non-preferential
                cmb = on_screen & highest_detection

                if np.count_nonzero(cmb) == 1:
                    # That's the one that works best: 
                    best_idx = int(np.where(cmb)[0][0])
                elif np.count_nonzero(cmb) == 0:
                    # There are no "optimal" detections. This means that only
                    # the level-1 detection, which by default is on_screen = True,
                    # has a full face. Let's use that. 
                    top_level_idx = np.where(det_levels == 1)[0]
                    if len(top_level_idx) == 1:
                        best_idx = int(top_level_idx[0])
                    else:
                        # There are multiple, suboptimal lower-level detections. 
                        # Just choose the one with the largest bounding box. 
                        best_idx = np.argmax(bbox_size)
                else:
                    # There are some highest-zoom detections that are fully on screen.
                    # Just pick the one with the largest bounding box.
                    candidates = np.where(cmb)
                    bbox_candidates = bbox_size[candidates]
                    big_box_idx = np.argmax(bbox_candidates)
                    best_idx = candidates[0][big_box_idx]

                # From the list of face_group, select the detection at best_idx
                # and put it on the deduplicated_faces list
                chosen_detection = face_group[best_idx]
                deduplicated_faces.append(chosen_detection)

        assert len(deduplicated_faces) == rank
        assert type(deduplicated_faces) == list
        assert type(deduplicated_faces[0]) == insightface.app.common.Face

        return deduplicated_faces