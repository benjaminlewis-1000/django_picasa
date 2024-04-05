#! /usr/bin/env python

import os
from face_manager.models import Person, Face

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
print(people)