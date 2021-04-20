#! /usr/bin/env python

from django.core.management.base import BaseCommand
from face_manager import models as face_models
from facenet_pytorch import MTCNN, InceptionResnetV1
from filepopulator import models as file_models
from io import BytesIO
from PIL import Image, ExifTags
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time
import torch


def get_embedding(highlight):

    image = (highlight - 127.5) / 128
    image = torch.Tensor(image)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    encoding = resnet(image.unsqueeze(0))
    encoding = encoding.detach().numpy().astype(np.float32)
    encoding = list(encoding[0])

    return encoding
    
def open_img_oriented(filename):

    try:
        image = PIL.Image.open(filename)
    except Exception as e:
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

    return image

def highlight_from_face(face):
    # Helper method to extract a 500x500 pixel highlight
    # image from the face. Uses the source_image_file
    # of the face to get the original image, extracts the
    # highlight given the face borders, and resizes. Returns
    # the face as a ContentFile, which is used in the 
    # save method for the person.highlight_image

    # Get the borders
    r_left = face.box_left
    r_right = face.box_right
    r_bot = face.box_bottom
    r_top = face.box_top

    # Make the borders a square
    top_to_bot = r_bot - r_top
    left_to_right = r_right - r_left
    lr_cent = r_left + left_to_right // 2
    tb_cent = r_top + top_to_bot // 2

    extent = min(top_to_bot, left_to_right) // 2
    r_left = lr_cent - extent
    r_right = lr_cent + extent
    r_top = tb_cent - extent
    r_bot = tb_cent + extent

    # Extract the source file, oriented. 
    source_file = face.source_image_file.filename
    image = open_img_oriented(source_file)
    # Crop out the image, resize, encode in ByteIO, etc.
    img_thmb = image.crop((r_left, r_top, r_right, r_bot))
    img_thmb = np.array(img_thmb)
    # print(img_thmb.shape)
    assert img_thmb.shape[0] == img_thmb.shape[1]
    img_thmb = cv2.resize(img_thmb, (160, 160))
    # Convert color space
    img_thmb = cv2.cvtColor(img_thmb, cv2.COLOR_BGR2RGB)
    img_thmb = np.moveaxis(img_thmb, 2, 0)

    return img_thmb

class Command(BaseCommand):
    
    def handle(self, *args, **options):
        faces = face_models.Face.objects.filter(face_encoding_512=None)
        print(len(faces))

        count = 0
        num_face = len(faces)
        for f in faces:
            try:
                t1 = time.time()
                highlight = highlight_from_face(f)
                t2 = time.time()
                enc = get_embedding(highlight)
                t3 = time.time()
                f.face_encoding_512 = enc
                super(face_models.Face, f).save()
            except Exception:
                print("No good")
                pass
            print(f"{count} / {num_face}, {count / num_face * 100:.3f}% done")
            count += 1

