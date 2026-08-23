# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A self-hosted Django + DRF photo library and face-tagging system ("django_picasa"). It indexes a photo tree on disk, extracts EXIF/GPS metadata and thumbnails, runs a face-detection/recognition pipeline (insightface/ONNX, formerly dlib/torch) to find and cluster faces across photos, and exposes everything through a REST API consumed by a separate frontend and a slideshow client. Background work (indexing, face extraction, classification) runs as Celery tasks on a schedule via `django-celery-beat`.

## Commands

- Run dev server: `python manage.py runserver`
- Run all tests: `python manage.py test`
- Run one app's tests: `python manage.py test filepopulator` / `python manage.py test face_manager` / `python manage.py test api`
- Run a single test case/method: `python manage.py test filepopulator.tests.ImageFileTests.test_thumbnails`
- Migrations: `python manage.py makemigrations && python manage.py migrate`
- Celery worker (required for indexing/face tasks to actually run): `celery -A picasa worker -l info`
- Celery beat (schedules the periodic tasks below): `celery -A picasa beat -l info`
- Docker (prod-like): `dockerize/docker-compose.yaml`; dev stack: `dockerize_dev/docker-compose.yaml`

There is no dedicated lint/format config in the repo; match existing style in the file you're editing.

## Environment / settings

`picasa/settings.py` branches heavily on `IN_DOCKER` and `PRODUCTION` env vars — non-Docker local dev uses hardcoded local paths/DB creds near the top of the `else` branch, Docker reads everything from env vars (`DJANGO_SECRET_KEY`, `DB_NAME`/`DB_USER`/`DB_PWD`, `DOMAINNAME`, `API_DOMAIN`, `FRONTEND_DOMAIN`, `TAILSCALE_HOST_IP`, `DOCKER_HOST_IP`, `PICASA_API_KEY`, `AUTHELIA_SECRET`, etc.). `DJANGO_SECRET_KEY` is required in both branches. When adding a new required setting, wire it into both branches or it will break local dev or Docker.

Auth in production goes through Authelia via `allauth`'s OIDC provider (see the config block and `picasa/adapters.py`); locally there's no SSO, so DRF's session/token/JWT auth classes carry local dev. The slideshow client bypasses normal auth entirely via `X-Slideshow-Key` / `?key=` checked against `SLIDESHOW_API_KEY` (see `api/permissions.py`'s `HasSlideshowKeyOrAuthenticated`) — any view meant to be reachable by the slideshow needs that permission class explicitly.

CORS/CSRF are locked to specific domains/regexes (`exploretheworld.tech` subdomains, plus the Tailscale/Docker host IPs) — don't loosen these to wildcard without asking, and remember any new frontend origin needs to be added to both `CORS_ALLOWED_ORIGIN_REGEXES` and `CSRF_TRUSTED_ORIGINS`.

## Architecture

Four Django apps, in a rough pipeline:

- **filepopulator** — the ingestion layer. `Directory` and `ImageFile` models track the photo tree; `scripts.py` walks `FILEPOPULATOR_SERVER_IMG_DIR` (`PHOTO_ROOT`), hashes files (pixel hash + file hash) to detect new/moved/changed/duplicate images, extracts EXIF/GPS, and generates big/medium/small thumbnails. Runs as scheduled Celery tasks (`tasks.py`): `populate_files_from_root`, `update_dir_dates`, `check_mod_dates`.
- **face_manager** — the ML pipeline. `Person` and `Face` models (the latter stores bounding boxes, 128-d and 512-d face encodings as Postgres `ArrayField`s, up to 5 weighted "possible identity" guesses, validation/rejection state). `face_extract_encode.py` (extraction) and `pyramidal_detector.py` (multi-scale detection with NMS) find faces in unprocessed `ImageFile`s; `assign_faces.py` (`faceAssigner`) classifies/clusters detected faces against known `Person`s. Runs as scheduled Celery tasks: `face_extraction`, `assign_faces`, `set_face_counts`. Has many one-off `management/commands/` for retraining, reassigning, XMP export, etc. — `management/commands/deprecated/` is dead code, don't build on it. `face_manager/dep/` is old/experimental model code (dlib, custom CNNs), also not part of the live pipeline.
- **api** — the DRF layer everything else talks to. `views.py` is one large file mixing standard `ModelViewSet`s (images, directories, faces, people) with bespoke `APIView`s for mobile tagging workflows (`ConfidentUnlabeledView`, `UnlabeledMobileInfo`, `ResetFace`, `MobileNameList`), JWT token endpoints, Authelia session state, and slideshow-facing endpoints. `permissions.py` defines the slideshow-key bypass used across the mobile/slideshow endpoints. Custom exception handling lives in `picasa/exceptions.py` (redirects unauthenticated API calls to login).
- **picasa** — project settings/URLs/celery app, plus the Authelia social-account adapter (`adapters.py`) that handles subdomain redirects after SSO login.

`train_classify` exists but is largely superseded by `face_manager`'s current insightface-based pipeline — check whether code there is still live before extending it.

The `image_face_extractor` git submodule (separate `faceTagging` repo) is referenced by older docs/scripts (`steps.txt`) as a standalone GPU-side face-processing server; the in-process `face_manager` pipeline is what's actually wired into Celery now.

## Data model notes

- `Face.declared_name` and the five `poss_identN` fields all use `on_delete=models.SET(get_default_blank_person)` — deleting a `Person` doesn't cascade-delete their faces, it reassigns them to the sentinel "no face assigned" person (`settings.BLANK_FACE_NAME`). That sentinel `Person` must always exist; `get_default_blank_person()` assumes it does rather than safely creating it (the fallback branch has bugs — don't rely on it running).
- `ImageFile.directory` uses `on_delete=models.PROTECT` — you cannot delete a `Directory` while it still has images; `Face.source_image_file` uses `CASCADE` — deleting an `ImageFile` deletes its faces.
- `Face.save()` and `Person.delete()`/`Face.delete()` do real validation and filesystem side effects (removing thumbnail files from disk) — don't bypass `save()`/`delete()` with `.update()` or raw queries when those invariants matter.
- `ImageFile.save()` unconditionally recomputes the pixel MD5 hash (`_generate_md5_hash()`, which fully decodes the image) on *every* save, not just creation — any `.save()` call on a row whose file has since become corrupted on disk will raise, not just initial ingestion of a bad file.
- Known open gaps (see `todos.txt`): no confirmed cascade behavior when an `ImageFile` is deleted vs. its associated faces in all code paths; some `settings.LOGGER.error("Need better handling on foreign_key")` markers indicate known-rough edges in `face_manager` tasks.

## Testing

There's a full test suite now (`api`, `face_manager`, `filepopulator`, `common`, and `picasa` itself — `train_classify` untouched). `picasa` isn't in `INSTALLED_APPS` (it's the project, not an app), so its tests run via dotted path rather than app label:
```
python manage.py test --exclude-tag=slow   # fast unit/model/API tests only
python manage.py test --tag=slow           # real ML inference tests (face_manager only)
python manage.py test picasa.tests common  # project-level + shared-util tests
```
`face_manager/test_face_cache.py` caches real `PyramidalDetector` output keyed on `sha256(image bytes) + sha256(pyramidal_detector.py source)`, so repeat runs against an unchanged image with an unchanged detector skip the CPU cost entirely (~4s → ~0.01s per image). Change either the image or the detector's code and the cache key changes automatically.

**Bootstrapping a fresh DB from scratch is currently broken**: `api/views.py` runs a module-level query (`Person.objects.filter(person_name='.ignore')[0]`) that assumes the `.ignore`/`.realignore`/`_NO_FACE_ASSIGNED_`/etc. `Person` rows already exist — nothing in the codebase creates them (no migration, no fixture). The live DB only has them because someone created them by hand at some point. Any new test DB (or a genuine fresh install) needs these seeded before the first request touches an `api/` URL — see `ensure_sentinel_people()` in `api/tests.py`.

**Dev/test infra lives outside this repo entirely**: a separate git worktree (isolated from whatever `picasa_api`/`db_picasa` are running live) with its own `db_picasa_dev`/`task_redis_dev`/`picasa_api_dev_test` Docker containers (plain `docker run`, not `dockerize_dev`'s compose file — that Dockerfile is stale/broken, missing its own `requirements.txt` and still installing dlib/`face-recognition` instead of insightface). Fixture data (500 sampled real photos, 5 known-corrupted JPEGs pulled from production logs with `NOTES.md`, a handful of `.heic` files, filepopulator's existing `test_imgs_filepopulate`) lives under `/mnt/fast_storage/appdata/django_picasa/test_suite/` on the host, separate from the `production/` and stale 2020-era `dev/` subtrees already there.

**Known bugs the test suite found and documents (not fixed, per instruction) — each is `@unittest.expectedFailure` with a comment at the point it's caught:**
- `api/views.py` `filteredImagesView.get()`: if query params are present but none are `people`/`year_start`/`year_end` (e.g. just `?key=...`), `p_query` stays `None` and `ImageFile.objects.filter(None)` raises `TypeError` instead of returning "all images" like the no-params case does.
- `face_manager/models.py` `Face.remove_poss_ident()` (used by `associate_person`/`set_possibles_zero`/`clear_person`): clears a `poss_identN` FK by poking `self.__dict__['poss_identN_id'] = None` directly instead of `self.poss_identN = None`. Under Django 6, `Model.save()` reconciles each cached forward-FK object back onto its attname column before writing, so the manually-nulled attname gets silently overwritten by the *still-cached* related object's pk again — `poss_identN` is never actually cleared. `reject_association()` doesn't have this bug (it uses real `self.poss_identN = None` assignment).
- `face_manager/face_extract_encode.py` `find_and_encode_faces()`: the `except Exception: ... continue` around image loading never sets `isProcessed = True`, so a file that fails to decode (corrupt JPEG) is retried by the scheduled face-extraction task forever, on every run, with no backoff or dead-lettering. Reproduced with 5 real corrupted files pulled from production logs (`face_manager/tests.py` `FaceExtractorCorruptedImageTests`).
- `api/views.py` `bulk_thread()`: bare `except: print(...)` with no `continue` when a `face_id` doesn't resolve — falls through to use the unset `face` variable, raising `UnboundLocalError`, silently swallowed by the caller. A stale/bad ID in a bulk-operation request just silently no-ops.
- `filepopulator/models.py` `Directory.average_date_taken()`/`beginning_date_taken()`: both use `timezone.utc`, removed from `django.utils.timezone` in the Django version this app now runs (6.0) — should be `datetime.timezone.utc`. Not hypothetical: the scheduled `filepopulator.update_dir_dates` Celery task has crashed with this exact `AttributeError` on every single run, confirmed via `docker logs picasa_api`, and never gets past the first `Directory` (no per-item try/except in `update_dirs_datetime()`), so directory date aggregation has been completely non-functional since the upgrade.
- `filepopulator/models.py` `ImageFile._generate_md5_hash()`'s except clauses catch `TypeError` and `PIL.Image.DecompressionBombError` but not the plain `OSError` a corrupted JPEG actually raises — `create_image_file()` crashes outright on a corrupted file rather than skipping gracefully; only survivable in practice because `add_from_root_dir()` wraps each file in its own try/except (so ingestion of *new* corrupt files is skipped, silently, forever — same retry-forever shape as the face-extraction bug above, just one layer earlier).
- `common/open_img_oriented.py`: its try/except only wraps the initial `PIL.Image.open()` call, which succeeds even for a truncated/broken JPEG (PIL parses the header lazily). The real decode error only surfaces later at `np.array(image)` (or a caller's own pixel access), unguarded — despite the function *looking* like it degrades gracefully (returns `None` on failure), a corrupted file actually raises an uncaught `OSError`. This is the actual origin point of the two retry-forever bugs above; pinned down at this layer in `common/tests.py`.

**Not a bug, just dead code worth knowing about**: `picasa/custom_cors.py`'s `LocalNetworkCorsMiddleware` is fully commented out of `MIDDLEWARE` in `settings.py` — not currently active. No tests were written for it since testing inactive code would be misleading; if it's ever re-enabled, write tests for it then.

## Planned work

- **HEIC support**: currently unsupported — `ImageFile.filename`'s `RegexValidator` and `create_image_file()`'s own extension check both only accept `.jpg`/`.jpeg`. iPhones increasingly deliver `.heic` natively (1,855 found under the live `PHOTO_ROOT`'s `aggregated/` dir alone). Sample fixture files for this are already pulled into `/mnt/fast_storage/appdata/django_picasa/test_suite/heic_images/`.
- **Dead JWT auth code after the Authelia migration**: auth now goes through Authelia/OIDC (see `picasa/adapters.py`, `ACCOUNT_ADAPTER`), but `rest_framework_simplejwt` is still wired up in full — `SIMPLE_JWT` settings, `TokenPairWithUsername`/`api/token/obtain/`/`api/token/refresh/`, `token_blacklist` in `INSTALLED_APPS`, `PyJWT` as a direct dependency. Worth an audit for what's actually still reachable (the slideshow client? a mobile app?) vs. leftover from before Authelia, since it's a second parallel auth system to reason about/keep secure if nothing uses it anymore.
- **Face clustering quality**: how `face_manager/assign_faces.py`'s `faceAssigner` clusters/matches detected faces against existing `Person`s hasn't been reviewed this round — flagged as a bigger task of its own, separate from the pipeline plumbing bugs already found.
- **Similar-image search**: find images that are visually similar (not just exact pixel-duplicate) — e.g. near-duplicate bursts, edited/re-exported versions of the same shot.
- **Faster/better image hashing for duplicate detection**: `filepopulator` currently does a full MD5 over raw decoded pixels (`ImageFile._generate_md5_hash()`) to catch *exact* duplicates. A perceptual hash (pHash/dHash/aHash) would likely be faster and could double as the foundation for the similar-image search above, rather than solving them separately.
- **Backend geocoding with a real service, not an offline heuristic**: GPS lat/lon is already captured (`gps_lat_decimal`/`gps_lon_decimal` via `gpsphoto`) but never reverse-geocoded to a place name. Wants a precise geocoding service rather than an offline/heuristic library — likely rate-limited (e.g. Nominatim's usage policy), so needs caching/backoff, not a call-per-request. Bonus: fall back to the nearest larger city/landmark/national park when there's no exact precise place (useful for photos taken somewhere remote).
- **Slideshow metadata overlay**: serve slideshow images with the photo's date (nicely formatted, not a raw timestamp) and location (feeds off the geocoding work above) alongside the image itself.
