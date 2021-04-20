#! /usr/bin/env python

from filepopulator import models as file_models
from face_manager import models as face_models
from django.core.management.base import BaseCommand
import editdistance
from time import sleep
import numpy as np

class Command(BaseCommand):
    
    def handle(self, *args, **options):

        exit = False
        while not exit:
            x = input("What is a part of a name you'd like to search? ")
            if x.lower() == 'exit':
                exit = True
                continue
            if x.lower() == 'restart':
                continue 
                
            people = face_models.Person.objects.filter(person_name__icontains=x)
            for i, p in enumerate(people):
                print(f'[1-{i}] {p.person_name}')

            x = input("Is there a second name to search? ")
            if x.lower() == 'restart':
                continue 
            if x.lower() != 'n':
                
                people2 = face_models.Person.objects.filter(person_name__icontains=x)
                for i, p in enumerate(people2):
                    print(f'[2-{i}] {p.person_name}')

            j = input("What is the index of the first person?")
            if j.lower() == 'restart':
                continue 
            if not '1-' in j:
                print("Error.")
            else:
                p1 = int(j.replace('1-', ''))

            j = input("What is the index of the second person?")
            if j.lower() == 'restart':
                continue 
            if not '2-' in j and not '1-' in j:
                print("Error.")
            else:
                list1 = False
                list2 = False
                if '1-' in j:
                    list1 = True
                else:
                    list2 = True
                p2 = int(j.replace('1-', '').replace('2-', ''))

            person1 = people[p1]
            if list1:
                person2 = people[p2]
            else:
                person2 = people2[p2]
            print(person1.id, person2.id)
            self.create_association(person1.id, person2.id)

        # print(people[0])

        # print(people.count())
        # idx = 614
        # people = people[idx:]
        # names = [p.person_name for p in people]
        # names_cmp = names.copy()
        # person_ids = [p.id for p in people]

        # removed_idcs = []

        # for i, n in enumerate(names):
        #     cmp_subnames = np.array(names_cmp[i+1:])
        #     ids_sublist = np.array(person_ids[i+1:])
        #     distances = []
        #     for c in cmp_subnames:
        #         dist = editdistance.eval(n.lower(), c.lower())
        #         distances.append(dist)

        #     distances = np.array(distances)
        #     dist_sort_idcs = np.argsort(distances)
        #     cmp_sort = cmp_subnames[dist_sort_idcs]
        #     ids_sort = ids_sublist[dist_sort_idcs]

        #     print(f"{i + idx} | Base person: {n}")
        #     for j in range(min(20, len(cmp_sort))):
        #         print(f"[{j}]: {cmp_sort[j]}")

        #     print('===================')
        #     valid_choice = False
        #     x = input("Choose a number or 'n' for 'None': ")
        #     while not valid_choice:
        #         if x.lower() == 'n':
        #             valid_choice = True
        #         else:
        #             try:
        #                 x = int(x)
        #                 valid_choice = True
        #                 base_id = int(person_ids[i])
        #                 associate_id = int(ids_sort[x])
        #                 print(base_id, associate_id)
        #                 self.create_association(base_id, associate_id)
        #             except:
        #                 x = input("Invalid numerical or 'n' choice. Choose again: ")
        #             # Now to get the associated ids:
        #     print('===================')
        #     print('===================')
        #     print('===================')

    def create_association(self, base_id, associate_id):
        base_person = face_models.Person.objects.get(pk=base_id)
        associated_person = face_models.Person.objects.get(pk=associate_id)
        print(base_person, associated_person)
        # Find out which one you want to be primary:
        print("Which of these faces do you want to be primary?")
        print(f'1) {base_person.person_name}')
        print(f'2) {associated_person.person_name}')
        x = ''
        while x not in ['1', '2']:
            x = input("Choose 1 or 2: ")

        # Get the list of Face objects for each
        if x == '1':
            base_faces = face_models.Face.objects.filter(declared_name__id=base_id)
            second_faces = face_models.Face.objects.filter(declared_name__id=associate_id)
            new_id = face_models.Person.objects.get(pk=base_id)
            old_id = associate_id
        else:
            base_faces = face_models.Face.objects.filter(declared_name__id=associate_id)
            second_faces = face_models.Face.objects.filter(declared_name__id=base_id)
            new_id = face_models.Person.objects.get(pk=associate_id)
            old_id = base_id
        print(base_faces.count())
        print(second_faces.count())
        print(new_id)
        for f in second_faces:
            assert f.declared_name.id == old_id
            # print(f.id, f.declared_name.id, f.written_to_photo_metadata)
            f.declared_name = new_id
            f.written_to_photo_metadata = False
            super(face_models.Face, f).save()

        # Delete the old person object. 
        old_person = face_models.Person.objects.get(pk=old_id)
        # Assert there are no faces linked to it. 
        old_num_faces = face_models.Face.objects.filter(declared_name__id=old_id).count()
        assert old_num_faces == 0
        old_person.delete()
