#! /usr/bin/env python

# One-time data migration: '.another_ignore' used to be a separate sentinel
# Person from '.ignore', created by the assign_faces classifier for
# low-confidence auto-suggestions. That split meant a face the classifier
# flagged as "probably nobody real" was never recognized by the UI's
# close_ignored bulk action, which only checked for '.ignore'/'.realignore'.
# settings.SOFT_IGNORE_NAME now points at '.ignore' directly, so this
# command folds any existing Face rows still pointing at '.another_ignore'
# over to '.ignore', then removes the now-empty sentinel Person.

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from face_manager.models import Face, Person

OLD_NAME = '.another_ignore'
NEW_NAME = '.ignore'
POSS_IDENT_FIELDS = [f'poss_ident{i}' for i in range(1, 6)]


class Command(BaseCommand):
    help = "Reassign Face rows referencing '.another_ignore' over to '.ignore', then delete '.another_ignore'."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true',
            help="Only print how many rows would be affected; don't write anything.",
        )
        parser.add_argument(
            '--yes', action='store_true',
            help="Skip the interactive confirmation prompt (needed for non-interactive/production runs).",
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        try:
            old_person = Person.objects.get(person_name=OLD_NAME)
        except Person.DoesNotExist:
            self.stdout.write(self.style.SUCCESS(
                f"No '{OLD_NAME}' Person found -- nothing to merge."
            ))
            return

        try:
            new_person = Person.objects.get(person_name=NEW_NAME)
        except Person.DoesNotExist:
            raise CommandError(
                f"'{NEW_NAME}' Person does not exist -- refusing to merge into a "
                "nonexistent target."
            )

        declared_qs = Face.objects.filter(declared_name=old_person)
        poss_counts = {
            field: Face.objects.filter(**{field: old_person}).count()
            for field in POSS_IDENT_FIELDS
        }

        self.stdout.write(f"declared_name={OLD_NAME}: {declared_qs.count()} face(s)")
        for field, count in poss_counts.items():
            self.stdout.write(f"{field}={OLD_NAME}: {count} face(s)")

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run -- no changes written."))
            return

        if not options['yes']:
            go_ahead = input(
                f"Reassign the above rows from '{OLD_NAME}' (id={old_person.id}) to "
                f"'{NEW_NAME}' (id={new_person.id}) and delete '{OLD_NAME}'? y/N: "
            )
            if go_ahead.lower() != 'y':
                self.stdout.write("Aborted.")
                return

        with transaction.atomic():
            declared_updated = declared_qs.update(declared_name=new_person)
            poss_updated = {}
            for field in POSS_IDENT_FIELDS:
                poss_updated[field] = Face.objects.filter(**{field: old_person}).update(
                    **{field: new_person}
                )
            old_person.delete()

        self.stdout.write(self.style.SUCCESS(
            f"Reassigned declared_name for {declared_updated} face(s)."
        ))
        for field, count in poss_updated.items():
            self.stdout.write(self.style.SUCCESS(f"Reassigned {field} for {count} face(s)."))
        self.stdout.write(self.style.SUCCESS(f"Deleted '{OLD_NAME}' Person."))
