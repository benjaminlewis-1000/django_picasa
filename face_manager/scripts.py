#! /usr/bin/env python

from .models import Person, Face
from django.conf import settings
from django.core.files.base import ContentFile
from facenet_pytorch import MTCNN, InceptionResnetV1
from filepopulator.models import ImageFile
from io import BytesIO
import cv2
import os
import image_face_extractor
import logging
import numpy as np
import shutil
import torch
import traceback
torch.backends.nnpack.enabled = False

def establish_server_connection():
    server_conn = image_face_extractor.ip_finder.server_finder(logger=settings.LOGGER)

    return server_conn

def establish_multi_server_connection():
    # server_conn = image_face_extractor.ip_finder.server_finder(logger=settings.LOGGER)
    server_conn = image_face_extractor.ip_finder_multi.server_finder(logger=settings.LOGGER)

    return server_conn


def get_inception_embedding(highlight):

    image = (highlight - 127.5) / 128
    image = torch.Tensor(image)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    encoding = resnet(image.unsqueeze(0))
    encoding = encoding.detach().numpy().astype(np.float32)
    encoding = list(encoding[0])

    return encoding

def placeInDatabase(foreign_key, face_data):

    if len(face_data) == 0:
        foreign_key.isProcessed = True
        foreign_key.save()
        return False

    for idx in range(len(face_data)):
        eachface = face_data[idx]
        name = eachface.name
        if name is None:
            face_data[idx].name = settings.BLANK_FACE_NAME
            name = settings.BLANK_FACE_NAME

        encoding = eachface.encoding
        if encoding is not None:
            encoding = list(encoding)
        square_face = eachface.square_face
        rect = eachface.rectangle
        r_left = rect.left
        r_right = rect.right
        r_bot = rect.bottom
        r_top = rect.top

        new_face = Face()
        
        # Set up and save the thumbnail.        
        # BGR to RGB by np.flip
        sq_thumb = eachface.square_face
        sq_thumb = cv2.cvtColor(sq_thumb, cv2.COLOR_BGR2RGB)
        
        if name is not None:
            # Find the person, if possible
            person_key = Person.objects.filter(person_name = name)
            if len(person_key) == 0:
                # No person was found. 
                new_person = Person(person_name = name)

                personal_thumbnail = cv2.resize(sq_thumb, (500, 500))
                # encode the image
                is_success, person_buff = cv2.imencode(".jpg", personal_thumbnail)
                # Save thumbnail to in-memory file as BytesIO
                person_byte_thumbnail = BytesIO(person_buff)

                person_thumb_filename = f'{foreign_key.pixel_hash}_{name}.jpg'
                person_byte_thumbnail.seek(0)
                # Load a ContentFile into the thumbnail field so it gets saved
                new_person.highlight_img.save(person_thumb_filename, ContentFile(person_byte_thumbnail.read())) 

                new_person.save()
                
                person_key = Person.objects.filter(person_name = name)
                assert len(person_key) == 1

            person_key = person_key[0]

            new_face.declared_name = person_key
            new_face.written_to_photo_metadata = True

        else:
            new_face.declared_name = None
            new_face.written_to_photo_metadata = False

        settings.LOGGER.debug(f"Found a person, {idx}")

        if encoding is not None:
            # Then write the encodings to the database. 

            ########################################
            # Extract inception features
            detected_face_nonsquare = eachface.square_face
            img_thmb = np.array(detected_face_nonsquare)
            assert np.abs(img_thmb.shape[0] - img_thmb.shape[1]) < 3, f"{img_thmb.shape[0]} != {img_thmb.shape[1]}"
            img_thmb = cv2.resize(img_thmb, (160, 160))
            # Convert color space
            img_thmb = cv2.cvtColor(img_thmb, cv2.COLOR_BGR2RGB)
            img_thmb = np.moveaxis(img_thmb, 2, 0)
            enc_512 = get_inception_embedding(img_thmb)
            ########################################

            new_face.face_encoding = encoding
            new_face.face_encoding_512 = enc_512
            new_face.reencoded = True

        settings.LOGGER.debug(f"Encoding done, {idx}")

        new_face.box_top = r_top
        new_face.box_bottom = r_bot
        new_face.box_left = r_left
        new_face.box_right = r_right

        new_face.source_image_file = foreign_key
        new_face.dateTakenUTC = foreign_key.dateTakenUTC

        sq_thumb_resize = cv2.resize(sq_thumb, settings.FACE_THUMBNAIL_SIZE)
        FTYPE = 'JPEG' # 'GIF' or 'PNG' are possible extensions
        # encode the image
        is_success, buffer_img = cv2.imencode(".jpg", sq_thumb_resize)
        # Save thumbnail to in-memory file as BytesIO
        temp_thumb = BytesIO(buffer_img)

        # sq_thumb = np.cv2.cvtColor(sq_thumb, cv2.COLOR_BGR2RGB)
        thumb_filename = f'{foreign_key.pixel_hash}_{foreign_key.file_hash}_face{idx}.jpg'

        temp_thumb.seek(0)
        settings.LOGGER.debug(f"New face object is populated, {idx}")

        # Load a ContentFile into the thumbnail field so it gets saved
        new_face.face_thumbnail.save(thumb_filename, ContentFile(temp_thumb.read())) #, save=False)

        temp_thumb.close()

        new_face.save()
        settings.LOGGER.debug(f"New face id is: {new_face.id}")

        foreign_key.isProcessed = True
        # Call the super because we just want to save the model,
        # not re-save the image. 
        super(ImageFile, foreign_key).save()
        
    return True


def populateFromImage(img_object, server_conn = None):

    if server_conn is None:
        server_conn = establish_server_connection()
    else:
        if server_conn.check_ip() is False:
            server_conn.find_external_server()

    filename = img_object.filename
    foreign_key = img_object

    changed_fk = False

    if foreign_key.isProcessed:
        return None, server_conn, changed_fk

    changed_fk = True

    # def face_from_facerect(self, filename):
    face_data = image_face_extractor.image_client.face_extract_client(filename, server_conn, logger=settings.LOGGER)
    
    placeInDatabase(foreign_key, face_data)
    print(f"Faces from image {filename} have been placed in database.")

    return face_data, server_conn, changed_fk 


# def populateFromImageMultiGPU(img_object, server_conn = None, server_ip = None, ip_checked=False):

#     filename = img_object.filename
#     foreign_key = img_object
    
#     if server_ip is None:
#         server_ip = 0
    
#     if server_conn is None:
#         server_conn = establish_multi_server_connection()
#         server_ip = 0
#     else:
#         if server_conn.check_ip(server_ip) is False:
#             server_conn.find_external_server()
#             server_ip = 0


#     changed_fk = False

#     if foreign_key.isProcessed:
#         return None, server_conn, changed_fk

#     changed_fk = True

#     if not os.path.isfile(filename): #could have moved
#         raise OSError(f"File {filename} not found.")

#     face_data = image_face_extractor.image_client_multi.face_extract_client(filename, server_conn, ip_address=server_ip, logger=settings.LOGGER, ip_checked = ip_checked)
#     print(f"Worked! IP was {server_ip}, length is {len(face_data)}, file is {filename}")
#     settings.LOGGER.debug(f"Worked! IP was {server_ip}, length is {len(face_data)}, file is {filename}")
#     placeInDatabase(foreign_key, face_data)
#     print(f"Faces from {filename} have been placed in database.")
#     settings.LOGGER.debug(f"Faces from {filename} have been placed in database.")

#     return face_data, server_conn, changed_fk 


def assignSourceImage(face_model, person_model):
    savePath = person_model.highlight_img.path

    shutil.copyfile(face_model.face_thumbnail.path, person_model.highlight_img.path)    
    
