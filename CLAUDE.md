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
- **api** — the DRF layer everything else talks to. `views.py` mixes standard `ModelViewSet`s (images, directories, faces, people) with JWT token endpoints, Authelia session state, and slideshow-facing endpoints; the bespoke `APIView`s for mobile tagging workflows (`ConfidentUnlabeledView`, `UnlabeledMobileInfo`, `ResetFace`, `MobileNameList`) were split out into `api/mobile_views.py` (2026-08-24) while fixing two bugs in them, since `views.py` had grown large. `permissions.py` defines the slideshow-key bypass used across the mobile/slideshow endpoints. Custom exception handling lives in `picasa/exceptions.py` (redirects unauthenticated API calls to login).
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

**Bootstrapping a fresh DB from scratch — fixed 2026-08-25** (see "Fixed bugs" for the full writeup): `api/views.py` used to run a module-level query (`Person.objects.filter(person_name='.ignore')[0]`) assuming the `.ignore`/`.realignore`/`_NO_FACE_ASSIGNED_`/etc. `Person` rows already existed, with nothing in the codebase creating them. Now fixed two ways together: the lookups are `SimpleLazyObject`-wrapped (defers the query past import time), and `face_manager/migrations/0003_seed_sentinel_people.py` creates the rows automatically as part of `manage.py migrate`. `ensure_sentinel_people()` in `api/tests.py` still exists as a defensive no-op for tests but is no longer the only thing creating these rows.

**Dev/test infra lives outside this repo entirely**: a separate git worktree at `/home/benjamin/git_repos/django_picasa_dev` on branch `backend_upgrade` (isolated from whatever `picasa_api`/`db_picasa` are running live) with its own `db_picasa_dev`/`task_redis_dev`/`picasa_api_dev_test` Docker containers (plain `docker run` on a dedicated `picasa_test_net` network, not `dockerize_dev`'s compose file — that Dockerfile is stale/broken, missing its own `requirements.txt` and still installing dlib/`face-recognition` instead of insightface). `picasa_api_dev_test` runs from the same `picasa_img:latest` image as production (so dependencies match exactly) with `sleep infinity` as its command — exec into it (`docker exec picasa_api_dev_test bash -c "cd /code && python manage.py test ..."`) rather than expecting it to serve anything. If these containers have been torn down, they're cheap to recreate: fresh `postgres:16-alpine`/`redis:7-alpine` containers, migrate, then seed the sentinel `Person` rows (see `ensure_sentinel_people()` in `api/tests.py` for exactly which ones and why). Real (non-synthetic) fixture data — 500 sampled real photos, the 5 known-corrupted JPEGs pulled from production logs with `NOTES.md`, `.heic` samples, filepopulator's real `test_imgs_filepopulate` — lives under `/mnt/fast_storage/appdata/django_picasa/test_suite/` on the host (used for local/manual runs, especially the `slow`-tagged real-inference tests); separate, small, git-committed *synthetic* equivalents live in `ci_fixtures/` in the repo itself, used only by CI (see below).
**Where things stand (as of 2026-08-25)**: all of the test/CI/dependency work below is committed to `backend_upgrade` and pushed to `origin/backend_upgrade`. PR #43 (`backend_upgrade` → `master`) is open to trigger the first real CI run — see "Planned work" for what's still outstanding before an actual merge/deploy. `master` separately got a small, unrelated CORS/CSRF fix + this file, plus (2026-08-25) the `.github/workflows/tests.yml` file itself, added directly so PR-triggered CI runs can fire at all (GitHub won't run a `pull_request`-triggered workflow the first time if the workflow file doesn't already exist on the base branch).

**Fixed bugs:**
- **`ResetFace.patch()` and `ConfidentUnlabeledView.get()` in `api/mobile_views.py`** (moved here from `api/views.py`, see the `api` architecture note above): `ResetFace.patch()` had no `return` statement, so DRF's `dispatch()` got `None` back instead of a `Response` and raised `AssertionError` — crashed on **every single call**, not an edge case. Fixed by returning a small JSON success body; also dropped a `@action(...)` decorator left over on it, which is a DRF-router-only decorator that does nothing on a plain `APIView`. `ConfidentUnlabeledView.get()` did `unlabeled[0].weight_1`/`unlabeled.last().weight_1` unconditionally (an unused sanity-check `assert`, not part of the response) — `unlabeled[0]` raised `IndexError` the moment there were zero unlabeled faces, which is the *goal* state of the tagging workflow, not a rare edge case. Fixed by just returning whatever ids exist, including none.
- **Open redirect in `picasa/adapters.py`'s `SubdomainRedirectAdapter.get_login_redirect_url()`** (found by a follow-up bug-hunt pass, not the original test-writing session): the post-login `?next=` check was `'facewire.exploretheworld.tech' in next_param` — plain substring containment, not host validation — so `next=https://evil.example/?x=facewire.exploretheworld.tech` passed and would have redirected a freshly-authenticated user's browser to an attacker-controlled host. Now parses `next_param` and validates the actual hostname against `^([a-zA-Z0-9_-]+\.)*exploretheworld\.tech$` (same pattern `CORS_ALLOWED_ORIGIN_REGEXES` already uses), trusting any real `exploretheworld.tech` subdomain rather than only the one hardcoded `facewire` case; a relative path (no host to spoof) is allowed through as-is, same as Django's own `next`-handling convention.
- **EXIF orientation handling was duplicated across three implementations, one of them wrong.** `common/open_img_oriented.py` only handled EXIF orientations 3, 6, 8 (via `rotate()`), silently doing nothing for 2, 4, 5, 7; `filepopulator/models.py`'s `ImageFile._init_image()` had its own separate, *correct* 8-value implementation (via `transpose()`); `face_manager/problem_photos.py` had a third copy (deprecated/dead — hardcoded a path to a different machine, unreferenced anywhere — deleted rather than merged). Consolidated into one shared `apply_exif_orientation(image, orientation)` in `common/open_img_oriented.py` (exported from `common/__init__.py`), using the correct 8-value logic; both `open_img_oriented()` and `ImageFile._init_image()` now call it instead of maintaining their own copies. Also explicitly treats orientation `0` (not a standard EXIF value, but present on ~1,090 images in the live library) the same as `1` — no rotation — rather than leaving it to fall through unhandled.

  **Real-world impact turned out to be tiny**, checked against the live DB before doing this work (204,685 total images): orientations 1/3/6/8 (already correct under the old code) cover 203,593 images; of the four previously-broken values, 2, 4, and 5 have **zero** occurrences in the whole library, and 7 has exactly **2** (`ImageFile` ids `315617` and `316082`, 1 and 2 `Face` rows respectively — detection did find faces on both, just at the wrong coordinates since the image was never rotated for them). Given that, no bulk backfill/reprocessing migration was built — once this lands on `master`, just reprocess those two specific images by hand (clear their `isProcessed`/existing `Face` rows and let `face_extraction` redo them; expect to re-tag the 3 faces on them, which is fast for 2 photos). See "Planned work" for the port-to-master TODO.

- `filepopulator` ingestion-side corrupted-file handling — `ImageFile._generate_md5_hash()`/`common/open_img_oriented.py`/`create_image_file()`/`add_from_root_dir()`. This turned out to be three separate decode-failure points, not one:
  1. `ImageFile._generate_md5_hash()`'s except clauses caught `TypeError`/`PIL.Image.DecompressionBombError` but not the plain `OSError` a corrupted JPEG actually raises. Now also catches `OSError`, falling back to `cv2.imread()` like the other branches (which is more tolerant of truncation than PIL and sometimes succeeds outright); if that also fails, raises one clear `OSError` instead of a downstream `AttributeError`.
  2. `_generate_thumbnail()` (called from `ImageFile.save()`) re-decodes the image via PIL to resize it and can raise `OSError` independently, *even when* `_generate_md5_hash()` above already succeeded via its `cv2.imread()` fallback — a second, separate failure point inside `instance_clean_and_save()` that needed its own handling.
  3. `common/open_img_oriented.py`'s try/except only wrapped the initial `PIL.Image.open()` call, which succeeds even for a truncated/broken JPEG (PIL parses the header lazily) — the real decode error surfaced later, unguarded, at `image.rotate()`/`np.array(image)`. Now the whole post-open block is wrapped, so it genuinely returns `None` on failure as documented (also fixes a real crash risk in `api/views.py`, which calls this directly to serve images).

  On top of catching these, failures are now recorded rather than silently retried forever: two new `ImageFile` fields, `image_load_failed`/`image_load_error` (added for the face_extraction fix below), are reused for a **previously-good photo that becomes unreadable** (its existing row is flagged via `.update()`, not `.save()`, to avoid re-triggering the same decode); a new model, `FailedImageFile` (filename/error/mtime), tracks a **file that's never successfully ingested** (no `ImageFile` row can exist for it — `.save()` needs a successful decode for width/height/thumbnails). `add_from_root_dir()` skips retrying a `FailedImageFile` entry whose mtime hasn't changed, but retries it (and clears the record on success) once the file's mtime does change. One more real gap found and fixed along the way: `create_image_file()`'s "pixel hash changed → delete old row, save replacement" branch used to delete the existing good row *before* attempting to save the replacement — if the new content turned out to be corrupted, that lost the row entirely with nothing to replace it. `instance_clean_and_save()` now returns `(success, error)` so this branch can save-then-delete instead, keeping (and flagging) the old row if the replacement fails.
- `face_manager/face_extract_encode.py` `find_and_encode_faces()`: the `except Exception: ... continue` around image loading used to never set `isProcessed = True`, so a file that fails to decode (corrupt JPEG) was retried by the scheduled face-extraction task forever, on every run. Fixed by setting `isProcessed = True` (stop the retry) and the `image_load_failed`/`image_load_error` fields above, so the failure is recorded instead of silently no-op'd. Uses `ImageFile.objects.filter(pk=...).update(...)` rather than `img_obj.save()` deliberately: `save()` unconditionally re-decodes the image via `_generate_md5_hash()`, which would raise an uncaught `OSError` on the same corrupted file right there in the failure handler. Verified against the 5 real corrupted files pulled from production logs (`face_manager/tests.py` `FaceExtractorCorruptedImageTests`).
- `filepopulator/models.py` `Directory.average_date_taken()`/`beginning_date_taken()`: used `timezone.utc`, an attribute removed from `django.utils.timezone` in the Django version this app now runs (6.0). Not hypothetical: the scheduled `filepopulator.update_dir_dates` Celery task crashed with this exact `AttributeError` on every single run, confirmed via `docker logs picasa_api`, never getting past the first `Directory` (no per-item try/except in `update_dirs_datetime()`), so directory date aggregation had been completely non-functional since the upgrade. Fixed with `pytz.utc` (already imported in this file) rather than `datetime.timezone.utc` — the module's own `from datetime import datetime` shadows the `datetime` module name with the class, so `datetime.timezone.utc` isn't reachable here.
- `face_manager/models.py` `Face.remove_poss_ident()` (used by `associate_person`/`set_possibles_zero`/`clear_person`): used to clear a `poss_identN` FK by poking `self.__dict__['poss_identN_id'] = None` directly instead of `self.poss_identN = None`, so Django 6's `Model.save()` FK-cache reconciliation silently restored the old value — `poss_identN` was never actually cleared. Now uses real `setattr()`/`getattr()`, matching how `reject_association()` always did it correctly. Also added `Face.NUM_POSSIBLE_IDENTITIES = 5` as the single source of truth (the `associate_person`/`set_possibles_zero` call-chains now loop over it instead of hardcoding `remove_poss_ident(1)` through `(5)`), plus a Django system check (`face_manager/apps.py`, `face_manager.E001`) that fails `manage.py check`/startup loudly if the model's actual `poss_identN`/`weight_N` field pairs ever stop matching that constant. Note: `set_possible_person()` and `reject_association()` still hardcode `5`/`range(1, 6)` via `eval`/`exec` — not touched, out of scope for this fix, would need a separate pass if `NUM_POSSIBLE_IDENTITIES` is ever actually changed.

**All 6 bugs originally found by the initial test-writing pass are now fixed** (the last two, below, plus the 4 above). None have been ported to `master`/deployed yet — see the TODOs in "Planned work".
- `api/views.py` `filteredImagesView.get()`: if query params were present but none were `people`/`year_start`/`year_end` (e.g. just `?key=...`), `p_query` stayed `None` and `ImageFile.objects.filter(None)` raised `TypeError` instead of returning "all images" like the no-params case does. Fixed by explicitly falling back to `ImageFile.objects.all()` when `p_query is None`, same as the no-params branch.
- `api/views.py` `bulk_thread()`: the bare `except: print(...)` around `Face.objects.get(id=face_id)` had no `continue`, so a bad/stale `face_id` let execution fall through to the operation branches with `face` either unbound (`UnboundLocalError` on the first list entry) or still holding the *previous* iteration's `Face` (silently operating on the wrong face for later entries) — either way swallowed by `background_bulk_processor()`'s blanket except. Fixed by catching `Face.DoesNotExist` specifically and adding the missing `continue`.

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

`dockerize/requirements.txt` on `master` is still the original, every entry an unbounded `>=` — untouched there deliberately, per the user's request to keep this work dev-only for now. On `backend_upgrade`, it's been trimmed and pinned: originally 15 packages removed as "zero references anywhere in the codebase" (`coloredlogs`, `dj-database-url`, `django-celery-beat`, `django-celerybeat-status`, `django-rest-framework` [a dead/unrelated stub package — not `djangorestframework`, which stays], `django-timezone-field`, `ExifRead`, `importlib-metadata`, `pgi`, `piexif`, `psycopg2-pool`, `python-dotenv`, `python-xmp-toolkit`, `SCons`, `twilio`), and every remaining package pinned `==` to the exact version that passed all 93 fast tests. **Correction (2026-08-26)**: `ExifRead`/`piexif` were wrongly included in that "zero references" list and had to be added back — that check only looked for direct imports in our own code, missing that `gpsphoto` (which we do use, for GPS EXIF extraction) imports both itself (`import exifread`, `from piexif import load, dump`) without declaring them in its own `setup.py`'s `install_requires` (empty/broken packaging on GPSPhoto's part). Invisible locally because `picasa_img:latest` already had both installed from before the trim — only surfaced once CI actually ran `pip install -r requirements.txt` into a genuinely clean environment, which is exactly the kind of gap a real CI run is for. Actually-zero-reference removals (the other 13) still stand. Deliberately pinned Django to `6.0.8`, not the newer `6.1` that `pip install --upgrade` offered — upgrading to 6.1 (with scipy bumped to 1.18.1 alongside it) made the test suite hang indefinitely partway through `filepopulator`'s duplicate-detection tests; root cause not confirmed (Django vs. scipy), not chased further, just avoided. If picking this back up: reproduce in a throwaway container (not `picasa_api_dev_test`), and getting a real stack trace will need `--cap-add=SYS_PTRACE` on the container so `py-spy dump` can attach (it couldn't, last time).

## Follow-up bug audit (2026-08-24)

After the original test-writing pass's 6 bugs were all fixed, a further audit pass covering
previously-unreviewed areas (`face_manager/assign_faces.py`, untouched `api/views.py` mobile
endpoints, `filepopulator/scripts.py`'s remaining functions, `picasa/adapters.py`,
`api/permissions.py`) found more. Working through these one at a time, at the user's request:

- [x] **Open redirect in `picasa/adapters.py`** — fixed, see "Fixed bugs" above.
- [x] **`ResetFace.patch()` and `ConfidentUnlabeledView.get()`** — fixed, see "Fixed bugs"
  above. Also split all 4 mobile-app-facing views out of `api/views.py` into
  `api/mobile_views.py` while touching them (they were the entire `/api/mobile/...` URL group).
- [x] `reject_association_app_api()` (`api/views.py`) — removed rather than fixed, per the
  user's call. Had the same unguarded-assert root cause as the already-fixed `close_assigned`
  bug (calls `Face.reject_association()` unconditionally, which asserts the person is a
  `poss_identN` candidate, crashing with an unhandled 500 if passed an actual `declared_name`
  instead) — but checking both frontend repos this project has access to (`dev_facewire`,
  `facewires_frontend`), neither one actually calls this endpoint or its
  `disassociate_patch_url` (only `dev_facewire`'s `CLAUDE.md` mentions it in passing, while
  explaining the *other*, already-fixed `close_assigned` bug). Confirmed dead code, so it was
  deleted rather than fixed. `UnlabeledMobileInfo` (`api/mobile_views.py`) no longer includes
  `disassociate_patch_url` in its response, since it pointed at this now-removed route.
  `Face.reject_association()` itself is untouched and still live — `bulk_thread()`'s
  `close_assigned` branch still calls it for the "decline a candidate" case.
- [x] `SOFT_IGNORE_NAME` mismatch — fixed by collapsing the two identities rather than teaching
  `close_ignored` about a second one. `.another_ignore` (created by the scheduled `assign_faces`
  task for low-confidence auto-suggestions) and `.ignore` (the sentinel a human assigns via
  `close_unassigned`) were separate `Person` rows, so `bulk_thread`'s `close_ignored` — which
  only recognized `.ignore`/`.realignore` — could never promote a classifier-suggested face to
  hard-ignore. `SOFT_IGNORE_NAME` now equals `.ignore` directly. Code fix ships on
  `backend_upgrade`; the data side (92,780 faces with `declared_name='.another_ignore'`, 115,410
  with `poss_ident1` set to it, per a real production count) needs the
  `merge_another_ignore_into_ignore` management command run against production — see "Planned
  work" — before/alongside deploying this.
- [x] Orphaned `Face` thumbnail files on every scheduled cleanup — fixed. `filepopulator/
  scripts.py`'s `delete_removed_photos()` deletes `ImageFile` rows whose file vanished from
  disk; `Face.source_image_file`'s `CASCADE` meant Django's bulk-SQL cascade delete skipped
  `Face.delete()`'s override (which removes the thumbnail file from disk), silently orphaning
  it. `ImageFile.delete()` now explicitly deletes each related `Face` first (invoking `Face`'s
  own `delete()` override) before deleting itself, fixing every instance-level `ImageFile`
  deletion call site at once (`delete_removed_photos()`, `create_image_file()`'s duplicate/
  hash-mismatch branches), not just this one. Note: `filepopulator/management/commands/
  cleanDB.py`'s `models.ImageFile.objects.all().delete()` is a separate, bigger version of the
  same root cause (a bulk queryset `.delete()` skips instance `delete()` entirely, no override
  can fix that) — not touched, since that command is a deliberate full-wipe dev tool where
  leftover files likely don't matter as much.
- [ ] `classify_unassigned()` array-sizing bug (`face_manager/assign_faces.py`) — a
  stale-sized zero-padded array can pollute a max-similarity calculation, and can raise
  `IndexError` in a specific combination (rejected candidates + `.another_ignore` in the
  rejected set). Compounded by `execute()`'s per-face error handling being commented out, so
  any exception here aborts the *entire* scheduled `assign_faces` run, not just one face.
- [x] Misleading log message in `check_file_mods()` (`filepopulator/scripts.py`) — fixed. Was
  logging `filename` (leftover from an earlier, unrelated loop) instead of `modfile` on
  failure. Cosmetic only, didn't affect behavior, just made debugging real failures misleading.
- [x] `MobileNameList` (`api/mobile_views.py`) was an unfinished stub returning hardcoded
  placeholder data (`['a','b','c','d']`) — fixed. Now queries real `Person` names, excluding
  the sentinel/ignore rows via `settings.IGNORED_NAMES`.

## Planned work

- **Fixed (2026-08-25): `api/views.py`'s sentinel `Person` lookups crashed app startup on any
  DB without `.ignore`/`.realignore`/`BLANK_FACE_NAME` rows already present.** Found while
  getting CI (PR #43) to actually run — a genuinely fresh, empty CI database hits this on the
  very first `manage.py test`/`manage.py check`, since `soft_ignore_person`/`hard_ignore_person`/
  `blank_person` were plain module-level queries evaluated at import time (URL resolution),
  before any test's sentinel-seeding has run. Production never noticed because those rows were
  seeded by hand once, long ago. Fixed by wrapping all three in `SimpleLazyObject`, deferring the
  query to first actual attribute access. Covered by `LazySentinelPersonTests` in `api/tests.py`.
  auto-creating them via a migration — see the "sentinel `Person` rows ... auto-create via a data
  migration" entry further down for how that was actually resolved the same day.
- **DONE (2026-08-26): `backend_upgrade` merged into `master` (PR #43) and deployed to the live
  `picasa_api` container.** This closes out every "port to master and deploy" TODO that had
  accumulated in this file — the mobile-views split + `ResetFace`/`ConfidentUnlabeledView` fixes,
  `reject_association_app_api()` removal, the `ImageFile.delete()` orphaned-thumbnail fix, the
  `picasa/adapters.py` open-redirect fix, `filteredImagesView`/`bulk_thread()` fixes, the EXIF
  orientation consolidation, the corrupted-image tracking fixes (both `face_extract_encode.py`
  and filepopulator ingestion sides), the `average_date_taken`/`beginning_date_taken` `pytz.utc`
  fix, the `close_assigned` fix, and the `check_file_mods()`/`MobileNameList` fixes — all are now
  live in production, not just tested on `backend_upgrade`. New migrations applied cleanly
  (`face_manager.0002_face_detected_age` needed a one-time `--fake` apply first — production
  already had that column under a lost migration name from before git-tracking existed; purely a
  one-off for this specific database, not something a fresh install or CI ever hits). No
  dependency rebuild needed since `picasa_img`'s installed versions already matched every pin
  exactly. The two known EXIF-orientation-7 images (`ImageFile` ids `315617`, `316082`) have been
  manually reprocessed — `isProcessed` cleared, old (wrong-coordinate) `Face` rows deleted,
  `face_extraction` re-ran and correctly redetected all 3 faces with proper bounding boxes; all 3
  are currently unassigned and need re-tagging by hand (one was previously tagged "Gwendolyn
  Lewis").
- **Frontend: "failed to open" image list.** Two sources to query, now live in production: a
  previously-good photo that's since become unreadable is flagged via
  `ImageFile.objects.filter(image_load_failed=True)` (its old thumbnails/metadata are kept, just
  flagged); a file that's never been successfully ingested at all shows up in
  `FailedImageFile.objects.all()` instead (no `ImageFile` row exists for these — one can't be
  created without a successful decode). The frontend should surface both so the user can go fix
  or remove the underlying files. Not started on the frontend side — noted here so it isn't lost.
- **Remove the file-lock (`settings.LOCKFILE`) mechanism in `add_from_root_dir()`
  (`filepopulator/scripts.py`).** It's a plain `os.path.isfile()` check with no
  wait/retry/timeout, and no cleanup on crash — if a run dies or gets killed (`kill -9`, OOM,
  container restart) mid-ingestion, the lockfile is left behind and every subsequent scheduled
  run silently no-ops (`"Locked!"` then returns) forever, with no alerting. Ran into stale
  leftover hung `manage.py` processes in `picasa_api_dev_test` this session (unrelated root
  cause — non-daemon background thread, already documented above — not this lock), which
  prompted noticing the lock file itself has the same fragility. Worth replacing with something
  that can't wedge itself: a DB-backed lock with a timeout/heartbeat, or just relying on Celery's
  own task-overlap prevention if the scheduled task doesn't already have it. Not started.
- **DONE: the `.another_ignore` → `.ignore` production merge (data side ran 2026-08-25; code
  side landed with the 2026-08-26 `backend_upgrade` merge/deploy).** Data: applied directly via
  `docker exec picasa_api python manage.py shell` (not the `merge_another_ignore_into_ignore`
  command file itself, since it only existed on `backend_upgrade` at the time and `master`'s
  checkout was the live bind-mounted container -- ran the same bulk-`.update()` logic inline
  instead, after a fresh `pg_dump` backup and a `--dry-run`-style count check). Reassigned 92,850
  faces' `declared_name` and 115,335 faces' `poss_ident1` from `.another_ignore` (id 2333, now
  deleted) to `.ignore` (id 1403, now at 103,317 declared_name / 115,335 poss_ident1). Code:
  `SOFT_IGNORE_NAME` now equals `.ignore` in production's live `settings.py`, so `assign_faces`
  no longer recreates `.another_ignore` on its next scheduled run, and `close_ignored` correctly
  recognizes classifier-suggested candidates. Fully resolved.
- **TODO: stop relying on manually-synced cached face-count columns on `Person`.**
  `Person.num_faces`/`num_possibilities`/`num_unverified_faces` are plain `IntegerField`s only
  kept in sync by `increment_assigned()`/`decrement_assigned()`/etc (called from `Face` model
  methods like `associate_person()`), or recomputed wholesale by the scheduled
  `face_manager.set_face_counts` task (`face_manager/tasks.py`'s `reset_task`) — never by a live
  query, so any code path that mutates `Face.declared_name`/`poss_identN` without going through
  those model methods (e.g. a bulk `.update()`) silently leaves the cached numbers wrong until
  someone happens to notice or the scheduled task next runs. This bit us directly: right after
  the `.another_ignore` → `.ignore` merge (2026-08-25), the underlying `Face` rows were correct
  but `.ignore`'s cached counters were stale (`10,467`/`0` instead of the real
  `103,317`/`115,335`), which is what `PersonListView` (`api/views.py`) actually serves to the
  frontend for non-blank-sentinel people. Worked around in the moment by manually queuing
  `set_face_counts` (`reset_task.delay()`) rather than looping and saving every `Person` by
  hand — but that's a hack, not a fix; the real problem is that these numbers require a separate
  sync step *at all*. Also folds in a second bug found while investigating this: `PersonSerializer`
  (`api/serializers.py`) declares `num_possibilities = serializers.SerializerMethodField()` but
  `get_num_possibilities` is commented out — any code path that actually serializes a `Person`
  through this serializer (confirmed via direct test: `PersonSerializer(p).data` raises
  `AttributeError: 'PersonSerializer' object has no attribute 'get_num_possibilities'`) crashes;
  `PersonViewSet` (the `/api/people/` DRF router endpoint) uses this serializer, so it's likely
  broken for any request that hits it, while `PersonListView` (`/api/person_list/`, hand-rolled,
  no serializer) is presumably what the frontend actually relies on instead. Real fix needs to
  address both symptoms of the same root cause together: either make `num_faces`/
  `num_possibilities`/`num_unverified_faces` genuinely live (a `SerializerMethodField`/annotated
  queryset computed on read, like `get_num_faces` already correctly does, no cached column at
  all), or keep the cached-column approach but make every mutation path — `associate_person()`,
  `remove_poss_ident()`, any future bulk operation — update it as a mandatory part of the same
  transaction, with `get_num_possibilities` implemented to match instead of missing. Not started.
- **Fixed (2026-08-25): the non-daemon background thread in `api/views.py`** (`work_thread` /
  `background_bulk_processor`) — turned out not to be just a local testing annoyance ("looks
  hung, isn't"). In CI, with no `--keepdb` and no one around to manually `kill` the leftover
  process, this actually broke the run: the thread's held-open DB connection made the test
  runner's post-run `DROP DATABASE test_picasa` fail (`OperationalError: database "test_picasa"
  is being accessed by other users`), and the whole job then hung indefinitely since Python won't
  exit while a non-daemon thread is alive — would have run until GitHub's runner timeout (up to
  6 hours) rather than actually completing. Fixed with `daemon=True` on the thread constructor,
  so it's killed automatically at interpreter exit instead of blocking it. That alone stopped the
  hang but not the underlying `DROP DATABASE` failure/exit code 1 — the thread still held an open
  DB connection (Django only auto-closes connections at the end of a normal request/response
  cycle, which this loop never participates in), just no longer blocking process exit. Fully
  fixed by calling `connections.close_all()` each time the loop goes idle (empty queue), so it
  never holds a connection indefinitely. Verified against a genuinely fresh, non-`--keepdb`
  database (matching CI exactly): 120 tests, `OK`, clean `exit 0`. A real Celery-task redesign
  might still be worth it for other reasons, but this specific failure mode is fully resolved.
- **Fixed (2026-08-25): sentinel `Person` rows (`.ignore`, `.realignore`, `BLANK_FACE_NAME`, etc)
  now auto-create via a data migration** (`face_manager/migrations/0003_seed_sentinel_people.py`),
  closing the "Bootstrapping a fresh DB from scratch is currently broken" gap for real, not just
  the import-time crash the `SimpleLazyObject` fix addressed. Found this was necessary while
  testing the lazy-object fix against a genuinely fresh (non-`--keepdb`) database: without it,
  `face_manager.tests.PersonModelTests` failed with `Person.DoesNotExist` (that test class never
  seeded its own sentinel rows — only `api/tests.py`'s `ApiTestCase` did), and a regression test
  of my own failed with an ID mismatch, because each `TestCase` class's own `ensure_sentinel_people()`
  call was creating its *own* throwaway copy inside a per-class transaction that rolls back
  afterward — meanwhile the module-level `SimpleLazyObject` in `api/views.py` caches whichever
  copy it resolved *first*, forever, so later test classes' freshly-created rows had different
  IDs than what was cached. The migration runs once, automatically, as part of `manage.py
  migrate` — before any test's transaction begins — so every test class (and a real fresh
  install) now shares the same permanent rows, matching how production actually behaves.
  `ensure_sentinel_people()` in `api/tests.py` is now effectively a no-op safety net (its
  `exists()` check short-circuits immediately) rather than the sole source of these rows.
- **Investigate `ImageFile.save()`'s unconditional MD5 rehash** (see "Data model notes" above) —
  it fully decodes the image and recomputes `_generate_md5_hash()` on *every* `.save()` call, not
  just creation. Worth checking whether anything calls `.save()` on existing rows somewhere hot
  (bulk operations, periodic tasks) where this is pure wasted CPU, and whether the hash could be
  computed once and skipped on later saves when the file's mtime/size haven't changed.
- **HEIC support**: currently unsupported — `ImageFile.filename`'s `RegexValidator` and `create_image_file()`'s own extension check both only accept `.jpg`/`.jpeg`. iPhones increasingly deliver `.heic` natively (1,855 found under the live `PHOTO_ROOT`'s `aggregated/` dir alone). Sample fixture files for this are already pulled into `/mnt/fast_storage/appdata/django_picasa/test_suite/heic_images/`.
- **Dead JWT auth code after the Authelia migration**: auth now goes through Authelia/OIDC (see `picasa/adapters.py`, `ACCOUNT_ADAPTER`), but `rest_framework_simplejwt` is still wired up in full — `SIMPLE_JWT` settings, `TokenPairWithUsername`/`api/token/obtain/`/`api/token/refresh/`, `token_blacklist` in `INSTALLED_APPS`, `PyJWT` as a direct dependency. Worth an audit for what's actually still reachable (the slideshow client? a mobile app?) vs. leftover from before Authelia, since it's a second parallel auth system to reason about/keep secure if nothing uses it anymore.
- **Face clustering quality**: how `face_manager/assign_faces.py`'s `faceAssigner` clusters/matches detected faces against existing `Person`s hasn't been reviewed this round — flagged as a bigger task of its own, separate from the pipeline plumbing bugs already found.
- **Similar-image search**: find images that are visually similar (not just exact pixel-duplicate) — e.g. near-duplicate bursts, edited/re-exported versions of the same shot.
- **Faster/better image hashing for duplicate detection**: `filepopulator` currently does a full MD5 over raw decoded pixels (`ImageFile._generate_md5_hash()`) to catch *exact* duplicates. A perceptual hash (pHash/dHash/aHash) would likely be faster and could double as the foundation for the similar-image search above, rather than solving them separately.
- **Backend geocoding with a real service, not an offline heuristic**: GPS lat/lon is already captured (`gps_lat_decimal`/`gps_lon_decimal` via `gpsphoto`) but never reverse-geocoded to a place name. Wants a precise geocoding service rather than an offline/heuristic library — likely rate-limited (e.g. Nominatim's usage policy), so needs caching/backoff, not a call-per-request. Bonus: fall back to the nearest larger city/landmark/national park when there's no exact precise place (useful for photos taken somewhere remote).
- **Slideshow metadata overlay**: serve slideshow images with the photo's date (nicely formatted, not a raw timestamp) and location (feeds off the geocoding work above) alongside the image itself.
- **Video support**: the pipeline currently assumes still images end-to-end — `ImageFile`'s filename validator/extension checks, thumbnailing, EXIF/GPS extraction, and the face-detection pipeline are all image-only. Adding video would need real planning: a distinct model (or a shared base) for video assets, a thumbnailing strategy (extract a representative frame, or several), whether/how face detection runs against video (sample frames vs. skip entirely), metadata extraction differences (video containers carry EXIF-equivalent metadata differently than JPEGs), and slideshow/API changes to serve a different media type. Not started — flagged here as a bigger feature needing a design pass, not a quick add.
