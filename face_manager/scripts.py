
from django.conf import settings

from filepopulator.models import ImageFile
import image_face_extractor

def populateFromImage(filename):

    # def face_from_facerect(self, filename):
    face_data = image_face_extractor.image_client.face_extract_client(filename)
    print(face_data)
        # print(source_image_file)