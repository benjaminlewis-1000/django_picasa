from face_manager.models import Face, Person
from face_manager.assign_faces import faceAssigner

names = ['David Redd', 'Leslie Williams ', 'Emma', 'Elder Tim Call', 'Luca Wilson']
people = list(Person.objects.filter(person_name__in=names))
people_ids = [p.id for p in people]

before_counts = {}
for p in people:
    before_counts[p.person_name] = Face.objects.filter(poss_ident1=p).count()

print("Loading encodings...", flush=True)
fa = faceAssigner()
fa.load_encodings()
print("Encodings loaded.", flush=True)

target_faces = list(Face.objects.filter(poss_ident1_id__in=people_ids))
print(f"Reclassifying {len(target_faces)} currently-proposed faces...", flush=True)

reverted = 0
kept = 0
for i, f in enumerate(target_faces):
    old_person_id = f.poss_ident1_id
    fa.classify_unassigned(f)
    f.refresh_from_db()
    if f.poss_ident1_id != old_person_id:
        reverted += 1
    else:
        kept += 1
    if (i + 1) % 200 == 0:
        print(f"  ...{i+1}/{len(target_faces)} reverted={reverted} kept={kept}", flush=True)

print(f"Done. reverted={reverted} kept={kept}", flush=True)
print("", flush=True)
for p in people:
    after = Face.objects.filter(poss_ident1=p).count()
    print(f"{p.person_name!r}: before={before_counts[p.person_name]} after={after}", flush=True)
