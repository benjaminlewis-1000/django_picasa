#! /usr/bin/env python

# Data migration: seeds the sentinel Person rows the rest of the app
# assumes already exist (BLANK_FACE_NAME, '.ignore', '.realignore', etc --
# see settings.IGNORED_NAMES). Nothing else in the codebase created these;
# the live DB only had them because someone created them by hand at some
# point. That meant a genuinely fresh install (or CI's always-empty
# database) crashed the moment anything touched them -- see CLAUDE.md's
# "Bootstrapping a fresh DB from scratch" note. Runs once, automatically,
# as part of `manage.py migrate` -- before any request or test transaction
# gets a chance to run, so every consumer shares the same permanent rows.

from io import BytesIO

from django.core.files.base import ContentFile
from django.conf import settings
from django.db import migrations


def _tiny_jpeg_bytes():
    from PIL import Image

    img = Image.new("RGB", (8, 8), color=(0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def seed_sentinel_people(apps, schema_editor):
    Person = apps.get_model("face_manager", "Person")

    for name in set(settings.IGNORED_NAMES):
        if Person.objects.filter(person_name=name).exists():
            continue
        person = Person(person_name=name)
        person.highlight_img.save(
            f"{name.strip('.') or 'sentinel'}_sentinel.jpg",
            ContentFile(_tiny_jpeg_bytes()),
            save=False,
        )
        person.save()


def noop_reverse(apps, schema_editor):
    # Deliberately not reversible: these rows are load-bearing (Face FKs
    # default to the blank sentinel via get_default_blank_person(), and
    # api/views.py resolves '.ignore'/'.realignore' at first use) -- there
    # is no safe automatic "undo" once real data may reference them.
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("face_manager", "0002_face_detected_age"),
    ]

    operations = [
        migrations.RunPython(seed_sentinel_people, noop_reverse),
    ]
