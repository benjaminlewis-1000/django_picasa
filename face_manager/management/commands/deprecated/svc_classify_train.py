#! /usr/bin/env python


from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Count
from django.db.models import Q
from face_manager.models import Person, Face
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pickle
import time


class Command(BaseCommand):
    
    def handle(self, *args, **options):

        min_num = settings.FACE_NUM_THRESH 
        names = Person.objects.annotate(c=Count('face_declared')) \
            .filter(c__gt=min_num) \
            .filter(~Q(person_name__in=settings.IGNORED_NAMES) ) # Exclude people in the IGNORED_NAMES list.
        # Count by number of related faces
             # Limit to faces > min_num

        name_vec = []

        for n in names:
            # print(n.person_name, n.id)
            name_vec.append([n.person_name, n.id])

        person_faces = Face.objects.filter(declared_name__in=names)

        enc_128 = [x.face_encoding for x in person_faces if x.face_encoding is not None] 
        enc_128_reshape = np.array(enc_128)

        # Label ids are unique ids from the database and are, in general, 
        label_128 = [x.declared_name.id for x in person_faces if x.face_encoding is not None] 
        # not sequential nor zero-indexed. So we need to fix that. 
        ids_128 = set(label_128)
        id_to_class_128 = dict(zip(list(range(len(ids_128))), ids_128))
        class_to_id_128 = dict(zip(ids_128, list(range(len(ids_128)))))
        label_128 = np.array([class_to_id_128[x] for x in label_128])


        assert len(set(label_128)) == max(set(label_128)) + 1
        assert len(label_128) == enc_128_reshape.shape[0]
        data_128 = [enc_128_reshape, label_128, name_vec, id_to_class_128, class_to_id_128]

        with open('/code/data_128.pkl', 'wb') as fh:
            pickle.dump(data_128, fh)


        # Do the same thing with 512 dimensional
        enc_512 = [x.face_encoding_512 for x in person_faces if x.face_encoding_512 is not None] 
        enc_512_reshape = np.array(enc_512)

        # Label ids are unique ids from the database and are, in general, 
        label_512 = [x.declared_name.id for x in person_faces if x.face_encoding_512 is not None] 
        # not sequential nor zero-indexed. So we need to fix that. 
        ids_512 = set(label_512)
        id_to_class_512 = dict(zip(list(range(len(ids_512))), ids_512))
        class_to_id_512 = dict(zip(ids_512, list(range(len(ids_512)))))
        label_512 = np.array([class_to_id_512[x] for x in label_512])


        assert len(set(label_512)) == max(set(label_512)) + 1
        assert len(label_512) == enc_512_reshape.shape[0]
        data_512 = [enc_512_reshape, label_512, name_vec, id_to_class_512, class_to_id_512]

        with open('/code/data_512.pkl', 'wb') as fh:
            pickle.dump(data_512, fh)

        # clf_small = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', C=0.99, class_weight='balanced', probability=True, verbose=True))
        # # print(X_small.shape)
        # s = time.time()
        # clf_small.fit(enc_128_reshape, label_128); print(time.time() - s)
        # _ = joblib.dump(clf_small, small_clf_file, compress=9)