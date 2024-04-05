#! /usr/bin/env python

import os
from face_manager.models import Person, Face
from filepopulator.models import ImageFile
from django.core.management.base import BaseCommand
from django.db.models import Q
import shutil

class Command(BaseCommand):

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, *args, **options):
                
        person = None

        people = []

        while person != "":
            person = input("List one person you would like to query: ")
            person = person.strip()
            people.append(person)
            print(person, type(person), len(person))

        year = input("What year? ")
        year = int(year)

        people.pop()

        # year = 2022
        # people = ["Benjamin Lewis", "Jessica Lewis", "Liam Lewis", "Nathaniel Lewis"]

        # crit_list = Q(declared_name__person_name=people[0])
        # crit_list2 = Q(declared_name__person_name=people[1])
        possibilities = None
        for p in people:
            crit_list = Q(declared_name__person_name=p)
            if possibilities is None:
                possibilities = Face.objects.filter(crit_list).values_list('source_image_file__id', flat=True) 
                possibilities = list(set(possibilities))
            else:
                p2 = Face.objects.filter(crit_list).values_list('source_image_file__id', flat=True) 
                p2 = list(set(p2))
                possibilities = set(possibilities).intersection(set(p2))
            print("UNION: ", len(possibilities))

        # Query by year:
        possibilities = list(possibilities)
        query = Q(dateTaken__year=year) & Q(id__in=possibilities)
        year_bound = ImageFile.objects.filter(query)
        print(year_bound.count())

        paths = list(year_bound.values_list('filename', flat=True).all())
        print(paths)

        for p in paths:
            shutil.copy(p, '/code/family_pics')
            # print(possibilities.count())


        # Get images of me
        # ImageFile.objects.filter()

        # print(crit_list)
        # possibilities2 = Face.objects.filter(crit_list2).values_list('id', flat=True) # & (criterion1 | criterion2 | criter
        # print((possibilities1 & possibilities2).count())
        # print(possibilities.count())

