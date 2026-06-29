# #! /usr/bin/env python

# NOTE: This is deprecated, the current iteration is in django_picasa under face_manager/pyramidal_detector.py

import sys
sys.path.append('/mnt/server_home/git_repos/django_picasa/face_manager')
from pyramidal_detector import PyramidalDetector
from insightface.app import FaceAnalysis
from PIL import Image, ExifTags
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import torch
import torchvision.ops.boxes as bops

def open_img_oriented(filename: str, as_numpy: bool):
    # Open an image, get its metadata from the EXIF tag,
    # orient it, and then return as a numpy array

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

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
    if as_numpy:
        image = np.array(image)
    return image


root_dir = '/mnt/NAS/Photos'
imgs = [
    # 'Pictures_In_Progress/2020/Adventures/20200829_131559.jpg',
    # "Pictures_In_Progress/2019/Baltimore Trip/DSC_1171.JPG",
    # 'Completed/Pictures_finished/Soph-Junior Years/Ward Opening FHE/RBL1208999_49.JPG',
    # 'Completed/Pictures_finished/Family Pictures/2011/2011 (12) December/Christmas 2011 (93).JPG',
    # 'Pictures_In_Progress/2019/Ben Work Trips/London and Brighton/2019-05-13 18.58.02.jpg', 
    # 'Completed/Pictures_finished/Family Pictures/2012/7-2012/July Trip/Washington DC (93).JPG',
    'Pictures_In_Progress/2022/nicholas_and_jessica/20220610_163606.jpg',
    ]

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

pyr = PyramidalDetector(detector = app, debug = True)

for ifile in imgs:
    full_path = os.path.join(root_dir, ifile)
    img = open_img_oriented(full_path, as_numpy=True)
    img2 = open_img_oriented(full_path, as_numpy=True)

    face_data = app.get(img)

    s = time.time()
    # deduplicated_faces, overlapping, box_edges = pyr.get(img)
    overlapping_detections, box_edges = pyr.find_raw_faces(img)
    print(f"Pyramidal takes {time.time() - s:.2f} sec")

    for det in overlapping_detections:
        pt1 = (int(det['bbox'][0]), int(det['bbox'][1]))
        pt2 = (int(det['bbox'][2]), int(det['bbox'][3]))
        cv2.rectangle(img, pt1, pt2, (100, 12, 150), 5)

    for box in box_edges:
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        cv2.rectangle(img, pt1, pt2, (1, 255, 50), 5)

    for det in face_data:
        pt1 = (int(det['bbox'][0]), int(det['bbox'][1]))
        pt2 = (int(det['bbox'][2]), int(det['bbox'][3]))
        cv2.rectangle(img2, pt1, pt2, (100, 255, 50), 5)


    bboxes = [det['bbox'] for det in overlapping_detections]
    bboxes = torch.tensor(np.array(bboxes))
    off_screen = [det['off_screen'] for det in overlapping_detections]
    level = [det['detect_pyr_level'] for det in overlapping_detections]

    plt.imshow(img)
    plt.show(block=False)
    plt.figure()
    plt.imshow(img2)
    plt.show(block=False)
    print(f"{full_path}: {len(face_data)}")

    self = pyr
    bboxes = [det['bbox'] for det in overlapping_detections]
    bboxes = torch.tensor(np.array(bboxes))
    iter_nums = np.array([det['iter_num'] for det in overlapping_detections])

    iou = self.iou_function(bboxes, bboxes)
    binary_iou = torch.gt(iou, self.iou_thresh).to(torch.float)

    # Remove the diagonal
    binary_iou_diag = binary_iou - torch.eye(binary_iou.shape[0])
    for ll in range(np.max(iter_nums) + 1):
        idcs = np.where(iter_nums == ll)[0]
        if len(idcs) > 0:
            mnidx = int(min(idcs))
            mxidx = int(max(idcs) + 1)
            binary_iou_diag[mnidx:mxidx, mnidx:mxidx] = 5


    img = open_img_oriented(full_path, as_numpy=True)
    for det in overlapping_detections:
        pt1 = (int(det['bbox'][0]), int(det['bbox'][1]))
        pt2 = (int(det['bbox'][2]), int(det['bbox'][3]))
        cv2.rectangle(img, pt1, pt2, (np.random.randint(255), np.random.randint(255), np.random.randint(255)), 5)
        
    plt.imshow(img)
    plt.show(block=False)