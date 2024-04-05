#! /usr/bin/env python

from face_manager.models import Person, Face
from django.core.management.base import BaseCommand


class Command(BaseCommand):

    def __init__(self):
        super(Command, self).__init__()


    # def add_arguments(self, parser):
    #     parser.add_argument('old_name', type=str, help='Old name, full or partial')

    def handle(self, *args, **options):
        old_name = input("What is the old name? ")
        print(old_name)
        # person_name = 'Alyssa'

        selected = False

        while not selected:
            rvals = Person.objects.filter(person_name__contains=old_name)
            print("Which of the following would you like to modify?")
            for ii, name in enumerate(rvals):
                print(f"({ii}) {name}")

            idx = int(input("Insert index number: "))
            go_ahead = input(f"You would like to modify {rvals[idx]}? y/N: ")
            print(go_ahead)

            if go_ahead.lower() == 'y':
                selected = True

        if selected:
            new_name = input("What is the new name? ")
            rvals[idx].person_name = new_name
            rvals[idx].save()
        
