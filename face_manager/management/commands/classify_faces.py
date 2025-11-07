#! /usr/bin/env python

from django.core.management.base import BaseCommand
from face_manager import assign_faces

class Command(BaseCommand):

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):

        classifier = assign_faces.faceAssigner(debug=True)
        # print("Clearing unassigned images...")
        # classifier.clear_unassigned_images()
        print("Classifying...")
        classifier.execute(redo_all = False)