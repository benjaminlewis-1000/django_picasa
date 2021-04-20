#! /usr/bin/env python

from django.conf import settings
from django.core.management.base import BaseCommand
from face_manager import face_classifier
from face_manager.models import Person, Face
import pickle
from django.db.models import Q

class Command(BaseCommand):
    
    # def add_arguments(self, parser):
    #     parser.add_argument('epoch', type=int)
    
    def handle(self, *args, **options):
        # Get all the unassigned Faces
        # epoch =options['epoch']
        # print(epoch)
        if False:
            criterion1 = Q(declared_name__person_name=settings.BLANK_FACE_NAME)
            criterion2 = ~Q(poss_ident1=None)
            unassigned = Face.objects.filter(criterion1 & criterion2)
            print(unassigned.count())
            # unassigned = unassigned[:15]

            for f in unassigned:
                f.poss_ident1 = f.poss_ident2 = f.poss_ident3 = f.poss_ident4 = f.poss_ident5 = None
                f.weight_1 = f.weight_2 =f.weight_3 = f.weight_4 = f.weight_5 = 0
                f.save()

            with open('/models/id_to_folder_map.pkl', 'rb') as ph:
                n = pickle.load(ph)
                net_class_to_id = {}
                for k in n.keys():
                    network_label = int(n[k].replace('/code/MISC_DATA/face_chips/', ''))
                    net_class_to_id[network_label] = k

            num_classes = len(net_class_to_id)
            print(net_class_to_id)


        # criterion_ign = Q(declared_name__person_name=settings.SOFT_IGNORE_NAME)
        with open("/code/epochs.txt", "r") as fh:
            data = fh.read()
            data = data.split('\n')
            if data[-1] == '':
                epoch = int(data[-2])
            else:
                epoch = int(data[-1])

        face_classifier.classify_unassigned_faces(batch_processing_size=128)

        criterion_ign = Q(declared_name__person_name__in=settings.IGNORED_NAMES)
        ignored = Face.objects.filter(criterion_ign)
        # print(ignored.count())
        total_todo = 127300
        num_batches = 500
        batch_size = int(total_todo // num_batches) + 1
        start_batch = epoch
        from_end = (num_batches - start_batch) * (batch_size)
        start = max(ignored.count() - from_end, 0)
        print(epoch, start)
        # ignored = ignored[start:]
        # num_ignored = len(ignored)
        print(ignored.count())

        for i in range(start_batch, num_batches):
            adj_idx = i - start_batch
            print(adj_idx)
            print(f"CLASSIFY BATCH {i}")
           
            with open("/code/epochs.txt", "a") as fh:
                fh.write(f'{i}\n')
            unassigned_subset = ignored[start + adj_idx * batch_size: start + (adj_idx+1) * batch_size]
            print(len(unassigned_subset), start + adj_idx * batch_size, start + (adj_idx+1) * batch_size)
            face_classifier.classify_unassigned_faces(do_ignore=True, unassigned_external = unassigned_subset)
