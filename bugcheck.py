#! /usr/bin/env python

import os
import numpy as np
from django.db.models import Count, Q, F
from face_manager.models import Person, Face
from face_manager.assign_faces import faceAssigner
import pickle
from django.conf import settings


# unassigned = Face.objects.filter(Q(declared_name__person_name=settings.BLANK_FACE_NAME))
# print(unassigned.count())
# Face.objects.filter(Q(declared_name__person_name=settings.BLANK_FACE_NAME)).update(rejected_fields = None)
# exit()
# classer = faceAssigner(debug=True)
# classer.load_encodings()

name = 'Meagan Van Katwyk'
name = 'Michael Williams'
name = 'Nicholas Lewis'
name = 'Nicole Hanks'
p = Person.objects.filter(person_name = name)[0]
person_id = p.id
print(p, person_id)

suggested_faces = Face.objects.filter(Q(poss_ident1__person_name=name))
ENCODINGS_PKL_FILE='/models/face_assign_preload.pkl'

firstname = name.split(' ')[0]
tmp_pkl = f'/tmp/{firstname}.pkl'

if os.path.exists(tmp_pkl):
	with open(tmp_pkl, 'rb') as ph:
		combo_dict = pickle.load(ph)
		person_candidate = combo_dict['candidate_dict']
		person_embedding = combo_dict['embedding_dict']
		person_norm = combo_dict['norm_dict']

else:
	with open(ENCODINGS_PKL_FILE, 'rb') as ph:
	    combo_dict = pickle.load(ph)
	    candidate_dict = combo_dict['candidate_dict']
	    embedding_dict = combo_dict['embedding_dict']
	    norm_dict = combo_dict['norm_dict']

	person_candidate = candidate_dict[person_id]
	person_embedding = embedding_dict[person_id]
	person_norm = norm_dict[person_id]

	combo_person_dict = {
		'candidate_dict': person_candidate,
		'embedding_dict': person_embedding,
		'norm_dict': person_norm,
	}


	with open(tmp_pkl, 'wb') as ph:
		pickle.dump(combo_person_dict, ph)

# print(norm_dict.keys())
print(type(person_norm), type(person_candidate), type(person_embedding))

for sf in suggested_faces:
	# Get the encoding
	# encoding = sf.face_encoding_512

	query_encoding = np.array(sf.face_encoding_512)
	query_encoding_norm = np.linalg.norm(query_encoding)
	# print(encoding)

	
	dot_product = np.dot(query_encoding, person_embedding)
	# assert len(dot_product) == len(cmp_encoding_norms)
	similarity = dot_product / (person_norm * query_encoding_norm)
	sim_max = np.max(similarity)
	sim_min = np.min(similarity)
	sim_5th = np.percentile(similarity, 5)
	sim_99th = np.percentile(similarity, 99)
	sim_995th = np.percentile(similarity, 99.5)
	print(sim_max, sim_99th, sim_min, sim_5th, sim_995th)
	# sf.update(rejected_fields = None)
	# sf.save()
	# classer.classify_unassigned(sf, debug_face_id=person_id)