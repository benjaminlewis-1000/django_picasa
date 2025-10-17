#! /usr/bin/env python

import os
import numpy as np
import insightface 
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import common
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
import time

def extract_faces(filename: str) -> dict:
    # Given a file name, use InsightFace to extract a dictionary of
    # people and 512-dimensional vector encodings.

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")

    img_numpy = common.open_img_oriented(filename)

    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
    app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU
    
    s = time.time()
    print(f"Starting encode at {s}")
    faces = app.get(img_numpy)
    print(f"Elapsed: {time.time() - s: .2f}")
    
    return faces