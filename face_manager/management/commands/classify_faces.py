#! /usr/bin/env python

from django.core.management.base import BaseCommand
from face_manager import assign_faces

class Command(BaseCommand):

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):

        DEBUG=False
        classifier = assign_faces.faceAssigner(debug=DEBUG)
        # classifier.load_encodings(reload_pkl_file = True)
        # exit()
        # if DEBUG:
        #     classifier.reset_possible_assignments()
        # print("Clearing unassigned images...")
        # classifier.clear_unassigned_images()
        classifier.reset_possible_assignments() ## OCCASIONAL
        print("Classifying...")
        classifier.execute(redo_all = False)
