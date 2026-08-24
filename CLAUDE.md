# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working conventions

**Don't touch `master` or this directory (`/home/benjamin/git_repos/django_picasa`) for exploratory/maintenance work.** This checkout is bind-mounted directly into the live `picasa_api` production container (`dockerize/.env`'s `DJANGO_FILES_ROOT` points here) — editing files here can affect what's actually running. Do this kind of work (tests, dependency upgrades, CI, bug investigation) in the `backend_upgrade` branch/worktree at `/home/benjamin/git_repos/django_picasa_dev` instead (see "Where things stand" below for what's already there). Only touch `master` directly for something the user explicitly asks to land on `master` right now.

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

**Dev/test infra lives outside this repo entirely**: a separate git worktree at `/home/benjamin/git_repos/django_picasa_dev` on branch `backend_upgrade` (isolated from whatever `picasa_api`/`db_picasa` are running live) with its own `db_picasa_dev`/`task_redis_dev`/`picasa_api_dev_test` Docker containers (plain `docker run` on a dedicated `picasa_test_net` network, not `dockerize_dev`'s compose file — that Dockerfile is stale/broken, missing its own `requirements.txt` and still installing dlib/`face-recognition` instead of insightface). `picasa_api_dev_test` runs from the same `picasa_img:latest` image as production (so dependencies match exactly) with `sleep infinity` as its command — exec into it (`docker exec picasa_api_dev_test bash -c "cd /code && python manage.py test ..."`) rather than expecting it to serve anything. If these containers have been torn down, they're cheap to recreate: fresh `postgres:16-alpine`/`redis:7-alpine` containers, migrate, then seed the sentinel `Person` rows (see `ensure_sentinel_people()` in `api/tests.py` for exactly which ones and why). Real (non-synthetic) fixture data — 500 sampled real photos, the 5 known-corrupted JPEGs pulled from production logs with `NOTES.md`, `.heic` samples, filepopulator's real `test_imgs_filepopulate` — lives under `/mnt/fast_storage/appdata/django_picasa/test_suite/` on the host (used for local/manual runs, especially the `slow`-tagged real-inference tests); separate, small, git-committed *synthetic* equivalents live in `ci_fixtures/` in the repo itself, used only by CI (see below).

**Where things stand (as of this session)**: all of the above test/CI/dependency work is committed to `backend_upgrade` (3 commits: migrations-tracking, the test suite itself, then CI+fixtures+dependency pins) and pushed to `origin/backend_upgrade` — deliberately kept off `master` for now at the user's request, not yet merged or PR'd. `master` separately got a small, unrelated CORS/CSRF fix + this file, pushed directly. **The CI workflow has not actually been exercised on GitHub yet** — it only triggers on `push`/`pull_request` targeting `master` (see `.github/workflows/tests.yml`), and `backend_upgrade` doesn't touch `master`, so pushing to `backend_upgrade` alone does not run it. Opening a PR from `backend_upgrade` into `master` would trigger it without merging anything, if a real end-to-end check is wanted before merging. The confirmed bugs listed below are deliberately **not fixed** — the user chose to build out test coverage and stabilize the environment first, bugs are next.

**Fixed bugs:**
- `face_manager/face_extract_encode.py` `find_and_encode_faces()`: the `except Exception: ... continue` around image loading used to never set `isProcessed = True`, so a file that fails to decode (corrupt JPEG) was retried by the scheduled face-extraction task forever, on every run. Fixed by setting `isProcessed = True` (stop the retry) and two new generic `ImageFile` fields, `image_load_failed`/`image_load_error`, so the failure is recorded instead of silently no-op'd — a frontend list of these is planned (see "Planned work") but out of scope here. Uses `ImageFile.objects.filter(pk=...).update(...)` rather than `img_obj.save()` deliberately: `save()` unconditionally re-decodes the image via `_generate_md5_hash()` (see bug below), which would raise an uncaught `OSError` on the same corrupted file right there in the failure handler. Verified against the 5 real corrupted files pulled from production logs (`face_manager/tests.py` `FaceExtractorCorruptedImageTests`). Note: this doesn't fix the ingestion-side corrupted-file handling (`ImageFile._generate_md5_hash()`/`common/open_img_oriented.py`, still open below) — a *new* corrupted file added to the photo tree still isn't tracked by `image_load_failed` until it's fixed too.
- `filepopulator/models.py` `Directory.average_date_taken()`/`beginning_date_taken()`: used `timezone.utc`, an attribute removed from `django.utils.timezone` in the Django version this app now runs (6.0). Not hypothetical: the scheduled `filepopulator.update_dir_dates` Celery task crashed with this exact `AttributeError` on every single run, confirmed via `docker logs picasa_api`, never getting past the first `Directory` (no per-item try/except in `update_dirs_datetime()`), so directory date aggregation had been completely non-functional since the upgrade. Fixed with `pytz.utc` (already imported in this file) rather than `datetime.timezone.utc` — the module's own `from datetime import datetime` shadows the `datetime` module name with the class, so `datetime.timezone.utc` isn't reachable here.
- `face_manager/models.py` `Face.remove_poss_ident()` (used by `associate_person`/`set_possibles_zero`/`clear_person`): used to clear a `poss_identN` FK by poking `self.__dict__['poss_identN_id'] = None` directly instead of `self.poss_identN = None`, so Django 6's `Model.save()` FK-cache reconciliation silently restored the old value — `poss_identN` was never actually cleared. Now uses real `setattr()`/`getattr()`, matching how `reject_association()` always did it correctly. Also added `Face.NUM_POSSIBLE_IDENTITIES = 5` as the single source of truth (the `associate_person`/`set_possibles_zero` call-chains now loop over it instead of hardcoding `remove_poss_ident(1)` through `(5)`), plus a Django system check (`face_manager/apps.py`, `face_manager.E001`) that fails `manage.py check`/startup loudly if the model's actual `poss_identN`/`weight_N` field pairs ever stop matching that constant. Note: `set_possible_person()` and `reject_association()` still hardcode `5`/`range(1, 6)` via `eval`/`exec` — not touched, out of scope for this fix, would need a separate pass if `NUM_POSSIBLE_IDENTITIES` is ever actually changed.

**Known bugs the test suite found and documents (not fixed, per instruction) — each is `@unittest.expectedFailure` with a comment at the point it's caught:**
- `api/views.py` `filteredImagesView.get()`: if query params are present but none are `people`/`year_start`/`year_end` (e.g. just `?key=...`), `p_query` stays `None` and `ImageFile.objects.filter(None)` raises `TypeError` instead of returning "all images" like the no-params case does.
- `api/views.py` `bulk_thread()`: bare `except: print(...)` with no `continue` when a `face_id` doesn't resolve — falls through to use the unset `face` variable, raising `UnboundLocalError`, silently swallowed by the caller. A stale/bad ID in a bulk-operation request just silently no-ops.
- `filepopulator/models.py` `ImageFile._generate_md5_hash()`'s except clauses catch `TypeError` and `PIL.Image.DecompressionBombError` but not the plain `OSError` a corrupted JPEG actually raises — `create_image_file()` crashes outright on a corrupted file rather than skipping gracefully; only survivable in practice because `add_from_root_dir()` wraps each file in its own try/except (so ingestion of *new* corrupt files is skipped, silently, forever — same retry-forever shape as the face-extraction bug above, just one layer earlier).
- `common/open_img_oriented.py`: its try/except only wraps the initial `PIL.Image.open()` call, which succeeds even for a truncated/broken JPEG (PIL parses the header lazily). The real decode error only surfaces later at `np.array(image)` (or a caller's own pixel access), unguarded — despite the function *looking* like it degrades gracefully (returns `None` on failure), a corrupted file actually raises an uncaught `OSError`. This is the actual origin point of the two retry-forever bugs above; pinned down at this layer in `common/tests.py`.

**Fixed this session (2026-08-24), breaking the "not fixed yet" pattern above because the
frontend (`dev_facewire`) hit it directly through its new undo/redo feature — not one of the
bugs the test suite above already found/documented:**
- `api/views.py` `bulk_thread()`'s `close_assigned` branch called `Face.reject_association()`
  unconditionally. That method only knows how to cross a candidate off a face's `poss_identN`
  "possible match" list, and asserts `current_person_id` is actually one of those candidates.
  That's correct when declining a proposed match, but `close_assigned` is also fired from
  "Remove from person" and (as of `dev_facewire`'s new undo/redo) undoing a `confirm_proposed` —
  both cases where the face is already **declared** to `current_person_id`, never a
  `poss_identN` entry, so the assert raised every time. That exception propagated out of
  `bulk_thread()` into `background_bulk_processor()`'s blanket `except Exception: print(...)`,
  silently swallowed — the queued job was just dropped, no error surfaced anywhere, and the
  face never actually moved. Fixed by checking which case it actually is: if
  `current_person_id` is a `poss_identN` candidate, still decline it via `reject_association`
  (unchanged); if it's the face's actual `declared_name`, reassign to `blank_person`
  (`_NO_FACE_ASSIGNED_`) via `associate_person()` instead — same mechanism `close_unassigned`
  already uses to reassign a face to `.ignore`. Covered by two new tests in
  `api/tests.py::FaceViewSetTests` (`test_bulk_close_assigned_on_declared_face_clears_name_tag`,
  `test_bulk_close_assigned_on_possible_match_still_declines_it`) that call `bulk_thread()`
  directly rather than going through the real `bulk_operation` HTTP endpoint + background
  queue/thread, since that path's own timing/DB-connection isolation isn't something a test
  should depend on. **This is only on `backend_upgrade`/`picasa_api_dev_test`, not `master`/the
  live `picasa_api` container** — the production API `dev_facewire`'s UI actually talks to
  (`picasa.exploretheworld.tech/api`) still has the original bug until this is ported to
  `master` and deployed. See `dev_facewire/CLAUDE.md`'s "Currently in progress / open" for the
  frontend-side note about this.
- Testing gotcha found while verifying the above (not itself an app bug, just a trap for
  future test runs): `api/views.py` starts a non-daemon background worker thread at *import*
  time (`work_thread = threading.Thread(target=background_bulk_processor); work_thread.start()`)
  running `while True: ...` forever. Once any test imports `api.views` (directly, or indirectly
  via the first request through DRF's URL routing), that thread keeps the whole `manage.py
  test` process alive even after every test has finished and results have printed — it just
  sits there, alive, never exiting on its own. Looks exactly like a hung test run (a `ps`
  snapshot shows low/flat CPU time, state `S`, blocked on a futex) when it's actually already
  done. Confirmed by killing the process after `Ran N tests ... OK` was already sitting in the
  (unflushed, pipe-buffered) output. Not chased further as an app fix - just know to check
  whether results already printed before assuming a `manage.py test` run is stuck, and expect
  to `kill` it rather than wait for a natural exit.

**Not a bug, just dead code worth knowing about**: `picasa/custom_cors.py`'s `LocalNetworkCorsMiddleware` is fully commented out of `MIDDLEWARE` in `settings.py` — not currently active. No tests were written for it since testing inactive code would be misleading; if it's ever re-enabled, write tests for it then.

## Dependencies

`dockerize/requirements.txt` on `master` is still the original, every entry an unbounded `>=` — untouched there deliberately, per the user's request to keep this work dev-only for now. On `backend_upgrade`, it's been trimmed and pinned: 15 packages with zero references anywhere in the codebase removed (`coloredlogs`, `dj-database-url`, `django-celery-beat`, `django-celerybeat-status`, `django-rest-framework` [a dead/unrelated stub package — not `djangorestframework`, which stays], `django-timezone-field`, `ExifRead`, `importlib-metadata`, `pgi`, `piexif`, `psycopg2-pool`, `python-dotenv`, `python-xmp-toolkit`, `SCons`, `twilio`), and every remaining package pinned `==` to the exact version that passed all 93 fast tests. Deliberately pinned Django to `6.0.8`, not the newer `6.1` that `pip install --upgrade` offered — upgrading to 6.1 (with scipy bumped to 1.18.1 alongside it) made the test suite hang indefinitely partway through `filepopulator`'s duplicate-detection tests; root cause not confirmed (Django vs. scipy), not chased further, just avoided. If picking this back up: reproduce in a throwaway container (not `picasa_api_dev_test`), and getting a real stack trace will need `--cap-add=SYS_PTRACE` on the container so `py-spy dump` can attach (it couldn't, last time).

## Planned work

- **TODO: port the `find_and_encode_faces()` corrupted-image fix (2026-08-24, adds
  `ImageFile.image_load_failed`/`image_load_error` + a migration) from `backend_upgrade` to
  `master` and deploy.** Fixes the retry-forever bug where corrupted images were reprocessed by
  the scheduled `face_extraction` task on every run, forever, with no record kept.
- **Frontend: "failed to open" image list.** Once images that fail to open/decode are tracked
  in the DB (see the bug #3/#6 fix work — corrupted-image retry-forever bugs in
  `face_manager`/`filepopulator`), the frontend should surface a list of them so the user can
  go fix or remove the underlying files. Not started, and explicitly out of scope for the
  backend fix itself — noted here so it isn't lost.
- **TODO: port the `average_date_taken`/`beginning_date_taken` `pytz.utc` fix (2026-08-24) from
  `backend_upgrade` to `master` and deploy.** The scheduled `filepopulator.update_dir_dates`
  Celery task has been crashing with `AttributeError` on every single run in production since
  the Django 6 upgrade (see "Fixed bugs" above) — directory date aggregation stays
  non-functional live until this ships.
- **TODO: port the `close_assigned` fix (2026-08-24) from `backend_upgrade` to `master` and
  deploy to the live `picasa_api` container.** Fixed and tested here (see "Fixed this session"
  above), but deliberately *not* ported/deployed yet, same as the rest of this branch's work —
  the frontend (`dev_facewire`) still talks to production and still has the original bug until
  this lands there. Don't consider this done until it's actually live.
- **HEIC support**: currently unsupported — `ImageFile.filename`'s `RegexValidator` and `create_image_file()`'s own extension check both only accept `.jpg`/`.jpeg`. iPhones increasingly deliver `.heic` natively (1,855 found under the live `PHOTO_ROOT`'s `aggregated/` dir alone). Sample fixture files for this are already pulled into `/mnt/fast_storage/appdata/django_picasa/test_suite/heic_images/`.
- **Dead JWT auth code after the Authelia migration**: auth now goes through Authelia/OIDC (see `picasa/adapters.py`, `ACCOUNT_ADAPTER`), but `rest_framework_simplejwt` is still wired up in full — `SIMPLE_JWT` settings, `TokenPairWithUsername`/`api/token/obtain/`/`api/token/refresh/`, `token_blacklist` in `INSTALLED_APPS`, `PyJWT` as a direct dependency. Worth an audit for what's actually still reachable (the slideshow client? a mobile app?) vs. leftover from before Authelia, since it's a second parallel auth system to reason about/keep secure if nothing uses it anymore.
- **Face clustering quality**: how `face_manager/assign_faces.py`'s `faceAssigner` clusters/matches detected faces against existing `Person`s hasn't been reviewed this round — flagged as a bigger task of its own, separate from the pipeline plumbing bugs already found.
- **Similar-image search**: find images that are visually similar (not just exact pixel-duplicate) — e.g. near-duplicate bursts, edited/re-exported versions of the same shot.
- **Faster/better image hashing for duplicate detection**: `filepopulator` currently does a full MD5 over raw decoded pixels (`ImageFile._generate_md5_hash()`) to catch *exact* duplicates. A perceptual hash (pHash/dHash/aHash) would likely be faster and could double as the foundation for the similar-image search above, rather than solving them separately.
- **Backend geocoding with a real service, not an offline heuristic**: GPS lat/lon is already captured (`gps_lat_decimal`/`gps_lon_decimal` via `gpsphoto`) but never reverse-geocoded to a place name. Wants a precise geocoding service rather than an offline/heuristic library — likely rate-limited (e.g. Nominatim's usage policy), so needs caching/backoff, not a call-per-request. Bonus: fall back to the nearest larger city/landmark/national park when there's no exact precise place (useful for photos taken somewhere remote).
- **Slideshow metadata overlay**: serve slideshow images with the photo's date (nicely formatted, not a raw timestamp) and location (feeds off the geocoding work above) alongside the image itself.
- **Video support**: the pipeline currently assumes still images end-to-end — `ImageFile`'s filename validator/extension checks, thumbnailing, EXIF/GPS extraction, and the face-detection pipeline are all image-only. Adding video would need real planning: a distinct model (or a shared base) for video assets, a thumbnailing strategy (extract a representative frame, or several), whether/how face detection runs against video (sample frames vs. skip entirely), metadata extraction differences (video containers carry EXIF-equivalent metadata differently than JPEGs), and slideshow/API changes to serve a different media type. Not started — flagged here as a bigger feature needing a design pass, not a quick add.
