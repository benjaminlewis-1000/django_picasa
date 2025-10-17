#! /usr/bin/env python

import cv2
import time
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=0)  # ctx_id=-1 for CPU, 0 for GPU

s = time.time()
file = 'biden.jpg'
img = cv2.imread(file)
print(img.shape)
faces = app.get(img)
print(f"Elapsed: {time.time() - s:.2f}")
