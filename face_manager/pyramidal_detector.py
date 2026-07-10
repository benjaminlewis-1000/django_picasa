#! /usr/bin/env python

from insightface.app import FaceAnalysis
import insightface 
import numpy as np
import torch
from collections import Counter
import torchvision.ops.boxes as bops
from torchvision.ops import nms as nms
import matplotlib.pyplot as plt

class PyramidalDetector():
    """
    A class that breaks an image into n x n rectangles, including the original
    image as a 1x1 square, and runs a FaceAnalysis detector on each level.
    It then de-duplicates detections across levels and rectangles to find an 
    'optimal' detection for each face. 
    """
    def __init__(self, detector: FaceAnalysis = None, debug: bool = False, iou_thresh = None):
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
        if iou_thresh is None:
            self.iou_thresh = 0.1
        else:
            self.iou_thresh = iou_thresh
        self.wholly_contained_pct_thresh = 0.9
        self.debug = debug

        # Set of pyramidal levels. E.g. 1 in the list is a 1x1 grid, 2 is a 2x2 grid, 3 is a 3x3 grid, etc.
        # We will run detections of all grids listed in this list, then
        # combine the results together with linear algebra. 
        self.cut_list = [1, 3]

    def box_intersection_pct(self, input_bboxes: torch.tensor, remove_diag: bool = False) -> torch.tensor:
        assert type(input_bboxes) == torch.Tensor
        assert len(input_bboxes.shape) == 2
        assert input_bboxes.shape[1] == 4

        n_bbox = len(input_bboxes)

        pct_intersect = torch.zeros((n_bbox, n_bbox))
        for row in range(n_bbox):
            for col in range(n_bbox):
                row_l, row_t, row_r, row_b = input_bboxes[row]
                col_l, col_t, col_r, col_b = input_bboxes[col]
                row_area = torch.abs(row_r - row_l) * torch.abs(row_t - row_b)
                # col_area = torch.abs(col_r - col_l) * torch.abs(col_t - col_b)
                intersect_l = torch.max(row_l, col_l)
                intersect_r = torch.min(row_r, col_r)
                intersect_t = torch.max(row_t, col_t)
                intersect_b = torch.min(row_b, col_b)
                if intersect_l > intersect_r or intersect_t > intersect_b:
                    intersect_pct = 0
                else:
                    intersect_area = torch.abs(intersect_l - intersect_r) * torch.abs(intersect_b - intersect_t)
                    intersect_pct = intersect_area / row_area
                assert intersect_pct <= 1
                # Row/column is the % of box in row that overlaps with box in col
                pct_intersect[row, col] = intersect_pct

        assert torch.all(torch.diag(pct_intersect) == 1)
        if remove_diag:
            pct_intersect = pct_intersect - torch.diag(torch.diag(pct_intersect))
        return pct_intersect

    def pct_overlap_winnow(self, intersect_pct_tensor: torch.Tensor, iter_nums: np.ndarray) -> torch.Tensor:
        assert type(intersect_pct_tensor) == torch.Tensor
        assert len(intersect_pct_tensor.shape) == 2
        assert type(iter_nums) == np.ndarray
        assert intersect_pct_tensor.shape[0] == intersect_pct_tensor.shape[1]
        assert intersect_pct_tensor.shape[0] == len(iter_nums)

        rows, cols = torch.where(intersect_pct_tensor > self.wholly_contained_pct_thresh)
        # print(rows, cols)
        bump_idcs = []
        affected_bbox = []
        for idx in range(len(rows)):
            crow = int(rows[idx])
            ccol = int(cols[idx])
            affected_bbox.append(crow)
            affected_bbox.append(ccol)
            row_in_col = intersect_pct_tensor[crow, ccol]
            col_in_row = intersect_pct_tensor[ccol, crow]
            if row_in_col > col_in_row:
                bump_idcs.append(crow)
            else:
                bump_idcs.append(ccol)

        bump_idcs = list(set(bump_idcs))
        affected_bbox = list(set(affected_bbox))
        bump_idcs.sort()
        # print(intersect_pct_tensor)
        # print(affected_bbox)
        # print(len(rows))
        # print(bump_idcs)
        if len(affected_bbox) > 0:
            assert len(bump_idcs) < len(affected_bbox)
        return bump_idcs

    def find_raw_faces(self, np_image: np.ndarray) -> list:
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
        box_edges = []

        iter_num = 0
        for cut_dims in self.cut_list:
            # Figure out the start and end row and columns for this. 

            if cut_dims == 1:
                # Just run the image
                faces_at_level = self.app.get(np_image)
                chip_h, chip_w, _ = np_image.shape
                for idx in range(len(faces_at_level)):
                    faces_at_level[idx]['detect_pyr_level'] = cut_dims
                    faces_at_level[idx]['iter_num'] = iter_num
                    faces_at_level[idx]['bbox'][0] = int(faces_at_level[idx]['bbox'][0]) # Left
                    faces_at_level[idx]['bbox'][2] = int(faces_at_level[idx]['bbox'][2]) # Right
                    faces_at_level[idx]['bbox'][1] = int(faces_at_level[idx]['bbox'][1]) # Top
                    faces_at_level[idx]['bbox'][3] = int(faces_at_level[idx]['bbox'][3]) # Bottom
                    # Even though it may technically go off-screen, this chip
                    # is always considered "on-screen" since there was no additional
                    # information to be had. 
                    faces_at_level[idx]['off_screen'] = False
                    
                overlapping_detections.extend(faces_at_level)
                iter_num += 1

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

                        box_edges.append((left_edge, top_edge, right_edge, bottom_edge))

                        # Detect on sub-image. Indicate whether the detection went off
                        # the edge of the chip, and adjust the bounding box to its
                        # proper location in the larger image. 
                        faces_at_level = self.app.get(chip_part)
                        for idx in range(len(faces_at_level)):
                            faces_at_level[idx]['detect_pyr_level'] = cut_dims
                            faces_at_level[idx]['iter_num'] = iter_num
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
                        iter_num += 1

        return overlapping_detections, box_edges

    def get(self, np_image: np.ndarray) -> list:
        if type(np_image) != np.ndarray:
            raise TypeError('Import image must be a numpy array.')
        if not bool(np.all(np.array(self.cut_list) >= 1)):
            raise ValueError("All values in the self.cut_list parameter must be >= 1.")
        if not np.array(self.cut_list).dtype == int:
            raise ValueError("All values in the self.cut_list parameter must be integers.")
        if 1 not in self.cut_list:
            raise ValueError("Need to have the value of 1 in self.cut_list")

        overlapping_detections, box_edges = self.find_raw_faces(np_image)

        bboxes = [det['bbox'] for det in overlapping_detections]
        bboxes = torch.tensor(np.array(bboxes))
        scores = [det['det_score'] for det in overlapping_detections]
        scores = torch.tensor(np.array(scores))
        iter_nums = np.array([det['iter_num'] for det in overlapping_detections])

        if len(overlapping_detections) == 0:
            return overlapping_detections

        # print(overlapping_detections[0].keys())
        # print(bboxes, scores)

        nms_out = nms(bboxes, scores, iou_threshold = self.iou_thresh).tolist()
        # print(nms_out)

        deduplicated_faces = [overlapping_detections[i] for i in nms_out]

        assert len(deduplicated_faces) == len(nms_out)
        assert type(deduplicated_faces) == list
        assert type(deduplicated_faces[0]) == insightface.app.common.Face
        return deduplicated_faces

        # if self.debug:
        #     return deduplicated_faces, overlapping_detections, box_edges
        # else:
        #     return deduplicated_faces
        # exit()
        """

        # Compute how much of each bounding box lies within
        # each other bounding box. Output is that a given
        # row, col value is how much the bounding box in the
        # row's index (in bboxes) lies in the bounding box of
        # the col's index (in bboxes).
        box_overlap_pcts = self.box_intersection_pct(bboxes, remove_diag = True)
        superfluous_idcs = self.pct_overlap_winnow(box_overlap_pcts, iter_nums)

        superfluous_idcs.sort()
        last_idx = len(bboxes)
        for pop_idx in superfluous_idcs[::-1]: # Reverse order
            assert last_idx > pop_idx # Sanity check
            overlapping_detections.pop(pop_idx)

        # Get bounding boxes again without superfluous overlaps
        bboxes = [det['bbox'] for det in overlapping_detections]
        bboxes = torch.tensor(np.array(bboxes))

        iou = self.iou_function(bboxes, bboxes)
        binary_iou = torch.gt(iou, self.iou_thresh).to(torch.float)

        # Remove the diagonal
        binary_iou_diag = binary_iou - torch.eye(binary_iou.shape[0])

        # print("BINARY", iou, binary_iou, box_overlap_pcts, superfluous_idcs)
        det_levels = np.array([f['detect_pyr_level'] for f in overlapping_detections])
        off_screen = np.array([f['off_screen'] for f in overlapping_detections])
        # print(det_levels, off_screen)
        # Binary IOU matrix should be symmetric, so assert that it is
        assert torch.all(binary_iou - binary_iou.T == 0)
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
        # print(binary_iou)
        # print(bboxes)
        # print(len(overlapping_detections))
        # print(unique_rows)
        # print(torch.sum(unique_rows))
        # print(iou)
        # print(superfluous_idcs)
        # # assert torch.sum(unique_rows) == len(overlapping_detections)
        # plt.imshow(unique_rows)
        # plt.show()

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
        # print(iou)
        # print(binary_iou)
        # print(rank)
        # print(bboxes)
        # print("TODO: put back assertion")
        eigen_count = Counter(large_eigens)
        sum_count = Counter(sums.tolist())
        
        assert eigen_count - sum_count == Counter(), f'Assumption that eigenvalues and rows sums match is inaccurate. {large_eigens} and {eigenvalues} VS {sums}'
        assert sum_count - eigen_count == Counter(), f'Assumption that eigenvalues and rows sums match is inaccurate. {large_eigens} and {eigenvalues} VS {sums}'

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

        if self.debug:
            return deduplicated_faces, overlapping_detections, box_edges
        else:
            return deduplicated_faces
        """