#! /usr/bin/env python

# import extract_picasa_faces as eff
from .XMPFace import XMPFace, Imagedata
from time import sleep
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import re
import shutil
import tempfile

def copyFile(src, dest):
    buffer_size = min(1024*1024,os.path.getsize(src))
    if(buffer_size == 0):
        buffer_size = 1024
    with open(src, 'rb') as fsrc:
        with open(dest, 'wb') as fdest:
            shutil.copyfileobj(fsrc, fdest, buffer_size)


def add_face_to_photo_xmp(photo_file, face_data, visualize = False):
    # Given the original photo and a dictionary of face bounding 
    # boxes, add it to the file XMP data. 
    assert 'name' in face_data
    assert 'top' in face_data
    assert 'left' in face_data
    assert 'right' in face_data
    assert 'bottom' in face_data


    temp_dir = tempfile.gettempdir()
    tmpfile = os.path.join(temp_dir, os.path.basename(photo_file) + (".%d.tmp" % random.randint(0,10000)))
    copyFile(photo_file,tmpfile)
    
    # Get a list of XMP faces already in the file
    # and initialize the XMP metadata.
    imgdata = Imagedata(photo_file)
    face = XMPFace(imgdata)

    # Set the image dimensions
    photo_pix = cv2.imread(photo_file)
    height, width, _ = photo_pix.shape
    face.setDim(width, height)

    # Get the number of existing faces
    xmp_faces = face.getFaces()
    num_faces = len(xmp_faces)

    # Read the file so we can extract height and width,
    # which are necessary for interpreting the XMP data.
    photo_pix = cv2.imread(photo_file)
    height, width, _ = photo_pix.shape

    # Create a numpy array that helps me calculate
    # overlap between XMP data and added face.
    overlap_calc = np.zeros((height, width))

    # For each face already in the XMP, read it and put it in the 
    # overlap calculation array. 
    for p in xmp_faces:
        xmp_left = int(p[0])
        xmp_top = int(p[1])
        xmp_width = int(p[2])
        xmp_height = int(p[3])
        xmp_right = xmp_left + xmp_width
        xmp_bottom = xmp_top + xmp_height

        assert xmp_left < xmp_right
        assert xmp_top < xmp_bottom

        # Add pixels to the overlap calculations. 
        overlap_calc[xmp_top:xmp_bottom, xmp_left:xmp_right] = 1

        if visualize:
            start_point = (xmp_left, xmp_top)
            end_point = (xmp_right, xmp_bottom)
            color = (255, 0, 0)
            thickness = 10
            cv2.rectangle(photo_pix, start_point, end_point, color, thickness) 

    # Get the points to add from the new face data.
    add_top = face_data['top']
    add_bottom = face_data['bottom']
    add_right = face_data['right']
    add_left = face_data['left']
    add_width = np.abs(add_left - add_right)
    add_height = np.abs(add_top - add_bottom)

    # Add to the overlap array. Any pixels that end up
    # being a 5 signify an overlap between existing XMP and 
    # the face to be added.
    overlap_calc[add_top:add_bottom, add_left:add_right] += 4

    # Calculate overlap percents, if any. 
    overlap_pxls = len(np.where(overlap_calc == 5)[0])
    add_pxls = add_width * add_height
    overlap_pct = overlap_pxls / (add_pxls)

    if overlap_pct > .3:
        print(f"Filename is {photo_file}")
        print(f"I don't know what to do in this situation. A face is already in XMP that overlaps with the added face. Overlap is {overlap_pct}") 
        return False

    if visualize:
        cv2.rectangle(photo_pix, (face_data['left'], face_data['top']), (face_data['right'], face_data['bottom']), (0, 100, 100), 5)
        cv2.imshow('Photo', photo_pix)
        cv2.waitKey(0)
        plt.imshow(overlap_calc)
        plt.show()

    # Set the face using exiv2
    face.setFace(add_left, add_top, add_width, add_height, face_data['name'], index=num_faces)
    # Save the file out. 
    face.save_file(photo_file)


    imgdata = Imagedata(photo_file)
    face = XMPFace(imgdata)

    xmp_faces = face.getFaces()
    num_faces_2 = len(xmp_faces)
    # print(num_faces_2, num_faces)

    if num_faces_2 <= num_faces:
        copyFile(tmpfile,photo_file)
        print("Number of faces is wrong.")
        return False

    assert num_faces_2 == num_faces + 1

    return True