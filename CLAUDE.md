# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working conventions

**Don't touch `master` or this directory (`/home/benjamin/git_repos/django_picasa`) for exploratory/maintenance work.** This checkout is bind-mounted directly into the live `picasa_api` production container (`dockerize/.env`'s `DJANGO_FILES_ROOT` points here) — editing files here can affect what's actually running. Do this kind of work (tests, dependency upgrades, CI, bug investigation) in the `backend_upgrade` branch/worktree at `/home/benjamin/git_repos/django_picasa_dev` instead (see "Where things stand" below for what's already there). Only touch `master` directly for something the user explicitly asks to land on `master` right now.

## Open TODOs (quick index)

A consolidated, easy-to-find list of everything still outstanding across this file — added
2026-09-04 after having to re-derive this list from a full-file grep sweep instead of it living
in one place. **Keep this section current going forward: add new open items here directly, not
only buried in a session's own narrative further down.** Full detail/context for each item is in
the dated write-up elsewhere in this file (search for a distinctive word from the bullet).

**Confirmed-live bugs, not yet fixed:**
- 29 duplicate `ImageFile` rows sharing the same filename (a different bug than the pixel_hash
  cross-path duplicate issue already fixed) — needs a decision on which row of each pair to keep.
- `find_and_encode_faces()`'s IOU-matching logic still has unhandled edge cases
  (`NotImplementedError`/`ValueError`/bare asserts) when an image's existing vs. newly-detected
  face counts diverge, and root-causing *why* those counts diverge is still open — but per-image
  **containment** was already fixed (commit `4613c84`, 2026-08-26, on both branches): a failure on
  one image no longer aborts the whole scheduled batch, just skips that image
  (`isProcessed=False`, retried later). Confirmed the 3 previously-stuck images (`99862`,
  `103837`, `108072`) are all `isProcessed=True` now — this entry is scoped down to just "find and
  fix the actual root cause," not "stops the whole pipeline," which is no longer true.
- `Person.num_faces`/`num_possibilities`/`num_unverified_faces` are manually-synced cached
  columns, not live queries — can silently drift stale if anything mutates faces outside the
  model's own methods. `PersonSerializer.get_num_possibilities` is also missing entirely (likely
  crashes `/api/people/` if that endpoint is ever actually hit).
- HEIC files with a non-1 EXIF orientation currently fail loudly instead of being handled — found
  via one real user photo, needs a design decision before extending the pipeline to apply it.

**Smaller tech debt:**
- `set_possible_person()`/`reject_association()` still hardcode `5`/`range(1, 6)` via `eval`/
  `exec` instead of using `Face.NUM_POSSIBLE_IDENTITIES` — fine until that constant ever changes.
- `ImageFile.save()` unconditionally re-decodes and rehashes the full image on *every* save, not
  just creation — flagged repeatedly as real wasted CPU on any hot path that re-saves existing
  rows, never actually fixed at the source (routed around instead, e.g. by `backfill_phash`).
- Django 6.1 upgrade is blocked: it made the test suite hang indefinitely partway through
  `filepopulator`'s duplicate-detection tests (Django vs. scipy 1.18.1 bumped alongside it, root
  cause never confirmed) — deliberately avoided, pinned at `6.0.8`.

**Open questions / follow-ups:**
- No automated "did last night's backup actually run" freshness check exists — the current
  restore-testing only validates a backup file once it's promoted into weekly retention, which
  says nothing about a night the backup silently never ran at all (this has already happened
  once, caught only because the user happened to notice a file hadn't shrunk).
- The `detected_age` birth-year-cutoff idea for face classification is tabled pending a visual
  sanity-check (real face thumbnails next to their `detected_age`) that was never actually done.
- A manual override tool for nearest-metro geocoding mismatches is wanted but not scoped (needs
  both a frontend UI and a backend endpoint/storage design).
- A looser classification threshold specifically for faces with `.ignore` already in their reject
  list was brainstormed but never validated or built.

**Bigger, deliberately unscoped features:**
- Slideshow metadata overlay (photo date + location shown alongside the image).
- Video support — the whole pipeline currently assumes still images end-to-end; needs a real
  design pass, explicitly not started.

**Frontend (out of scope for this repo — no visibility into that codebase from here):**
- "Mark image for deletion" button for the slideshow.
- "Failed to open" image list surface (backend data — `image_load_failed`/`FailedImageFile` — is
  ready; frontend work never started).
- Grouped-review UI for `Face.verification_cluster_group` (backend/data side is live).

**DONE (2026-09-04): the two persistently-failing tests, fixed.** Both were real test bugs, not
code bugs:
- `common.tests.OpenImgOrientedTests.test_corrupted_image_returns_none` hardcoded a filename
  (`truncated_a.jpg`) that matched neither the real local fixture set (5 real corrupted JPEGs
  pulled from production logs) nor CI's own synthetic `ci_fixtures/corrupted/` — always a
  `FileNotFoundError`. Fixed by picking whatever's actually present in `/photos/corrupted/`
  dynamically (`sorted(os.listdir(...))[0]`), matching the established convention other tests in
  this file already use for exactly this reason.
- `face_manager.tests.FaceModelTests.test_face_encoding_512_stores_at_float32_precision` compared
  the DB-round-tripped value against `float(np.float32(x))` for exact double equality --
  Postgres's `real` -> text -> Python float round trip (via psycopg2) doesn't necessarily
  reproduce the exact same double bit pattern as computing the float32 narrowing directly in
  Python (observed: `0.12345679` back from the DB vs `0.12345679104328156` computed directly).
  Fixed by narrowing both sides to `np.float32` before comparing, which collapses that harmless
  text-precision noise while still genuinely verifying the stored value lost precision down to
  float32 (the actual point of the test).
Full fast suite: **317/317 passing**, first fully-clean run this session.

**Stale — pruned from this file's later brainstormed-ideas list (2026-09-04):**
- "Similar-image search" / "faster image hashing for duplicate detection" — superseded:
  `backfill_phash`, `backfill_similarity`, and a live `filepopulator.find_similar_images`
  scheduled task already exist and do this.
- "Face clustering quality... hasn't been reviewed this round" — confirmed good for now by the
  user (2026-09-04); superseded by the much deeper outlier-rejection/gallery-size-adaptive-
  threshold investigation elsewhere in this file.

**Backend geocoding — implemented, just needs test coverage.** Nominatim-based reverse geocoding
was fully backfilled and runs on a schedule (`filepopulator.geocode_new_images`), but per the
user (2026-09-04) has no test exercising it yet. Add real test coverage for the geocoding path
(likely mocking the Nominatim HTTP call, given its rate-limit policy) before considering this
fully done.

**DONE (2026-09-04): stripped out the legacy `rest_framework_simplejwt` auth path.** The user
believed the external client project that depended on it (`/api/token/obtain/`,
`/api/token/refresh/`) had since dropped that dependency. Checked production logs before touching
anything: only 3 hits ever (within available log retention) to `/api/token/obtain/`, all on
2026-08-28 (about a week prior), all `Unauthorized` (failed auth attempts, not successful logins),
zero hits ever to `/api/token/refresh/`, and nothing at all in the week since — supported removing
it. **Important distinction preserved**: `PyJWT` itself (the `import jwt` package) is NOT related
to `rest_framework_simplejwt` and was correctly left alone — it's a separate, actively-live
dependency used by `api/authentication.py`'s `AutheliaOIDCAuthentication` (validates the
PhotoVerify mobile app's Authelia OIDC bearer tokens, RS256/JWKS). Removed: `rest_framework_
simplejwt`/`rest_framework_simplejwt.token_blacklist` from `INSTALLED_APPS`; `JWTAuthentication`
from `REST_FRAMEWORK['DEFAULT_AUTHENTICATION_CLASSES']`; the whole `SIMPLE_JWT` settings dict;
`TokenPairSerializer` (`api/serializers.py`) and `TokenPairWithUsername` (`api/views.py`); the
`token/obtain/`, `token/obtain` (redirect), and `token/refresh/` URL patterns (`api/urls.py`,
along with the now-unused `RedirectView` import); the 3 tests exercising the old endpoint
(`AuthenticationTests.test_token_obtain_with_valid_credentials`/
`test_token_obtain_with_bad_credentials_rejected`/`test_jwt_access_token_authenticates_requests`
in `api/tests.py`) and 1 in `picasa/tests.py`
(`test_jwt_signing_key_is_the_django_secret_key`); `djangorestframework-simplejwt==5.5.1` from
`dockerize/requirements.txt` (`PyJWT==2.13.0` and `cryptography` both kept — still real
dependencies, unrelated to this removal). Two stale comments in `api/authentication.py` that
referenced the old path as a "fallback" were also updated. **Deliberately not done**: the
`token_blacklist` app's DB tables were left in place rather than dropped (removing an app from
`INSTALLED_APPS` doesn't require dropping its tables, and there's no urgency); the Docker image
wasn't rebuilt to actually uninstall `djangorestframework-simplejwt` from `site-packages` (harmless
now that nothing imports it, just present-but-unused — a normal image rebuild whenever one next
happens will pick up the trimmed `requirements.txt`). Full fast suite: 317 tests total (4 fewer
than before, as expected from the removed tests), 315/317 passing — same 2 pre-existing,
unrelated failures as always.

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
- [ ] **Same anti-pattern, different task: `find_and_encode_faces()`'s IOU-matching branch
  (`face_manager/face_extract_encode.py` lines ~240-394) can silently abort the entire
  `face_extraction` run.** Found 2026-08-26 while investigating why the live "num images" vs.
  "num processed" stats were off by 3 after a manual `face_extraction` run. Root cause: when an
  `ImageFile` already has existing `Face` rows and this run's detector finds a different count
  (e.g. `FastFoto_0025.jpg` had 9 existing faces but only 5 were redetected), the mismatch is
  handled by IOU (bounding-box overlap) matching logic that has several genuinely unhandled edge
  cases left as hard failures by the original author — `raise NotImplementedError("Not
  one-to-one match")`, `raise ValueError("No overlapping detected and existing boxes...")`, plus
  a couple of bare `assert`s. None of this section is wrapped in the function's own try/except
  (which only covers image loading + detection, not the matching logic afterward), so any of
  these raises propagates all the way up. It's then caught by a **bare `except:` in
  `face_manager/tasks.py`'s `process_faces()`** (the actual Celery task for
  `face_manager.face_extraction`), which silently logs a DEBUG-level message and lets the task
  report "succeeded" — so there's no visible error anywhere. Whatever image was mid-processing
  never gets `isProcessed=True`, and critically, **every other unprocessed image still queued in
  that same run's random-ordered batch never gets attempted either**, since the whole loop
  aborts right there — that's the source of the "off by 3" discrepancy (one image crashed the
  matching logic, two more simply never got their turn). These are valid images that should
  successfully process — this needs the IOU-matching logic itself examined (why did the
  existing/detected face counts diverge, and what should actually happen in each unhandled
  case), not just a try/except slapped around it. Not fixed yet — documented per the user's
  request to investigate the actual IOU logic before deciding a fix. The 3 affected images from
  this run (`ImageFile` ids `99862`, `103837`, `108072`) remain unprocessed
  (`isProcessed=False`, `image_load_failed=False`) until this is resolved.
- [x] Misleading log message in `check_file_mods()` (`filepopulator/scripts.py`) — fixed. Was
  logging `filename` (leftover from an earlier, unrelated loop) instead of `modfile` on
  failure. Cosmetic only, didn't affect behavior, just made debugging real failures misleading.
- [x] `MobileNameList` (`api/mobile_views.py`) was an unfinished stub returning hardcoded
  placeholder data (`['a','b','c','d']`) — fixed. Now queries real `Person` names, excluding
  the sentinel/ignore rows via `settings.IGNORED_NAMES`.

## Planned work

**DONE (2026-09-04): investigated user-reported "near-exact overlap" duplicate faces; found and
fixed a real race condition, plus a new open TODO.** Started from the user noticing duplicate/
near-duplicate detected faces in the frontend. Quantified against real production data (pairwise
IOU over all images with 2-8 faces, ~566k pairs checked): 3,745 same-image Face-row pairs with
IOU > 0.9 (nearly all exactly 1.0 -- pixel-identical boxes). Two distinct causes found, confirmed
via the user's own two example faces (1060610/1076848) plus a broader id-delta analysis:
- **Root cause #1, ~72% of pairs (2,681), FIXED: a real race condition in
  `find_and_encode_faces()` (`face_manager/face_extract_encode.py`)**, not a pyramid-detector/NMS
  bug -- `pyramidal_detector.py`'s own `nms(iou_threshold=0.1 or 0.3 depending on caller)` looked
  correctly aggressive and wasn't implicated. The real gap: `find_and_encode_faces()` pulls
  `ImageFile.objects.filter(isProcessed=False)` and only marks `isProcessed=True` *after* fully
  processing an image -- nothing claims the row up front. Two concurrent invocations (from ANY
  entry point) processing the same never-before-processed image would each see `n_existing=0`,
  each run detection independently (deterministic, so pixel-identical boxes), and each call
  `add_new_face()` -- producing exactly the doubled rows found. The only existing guard was in the
  Celery task wrapper (`tasks.py`'s `process_faces()`), checking `celery_app.control.inspect().
  active()` -- a classic check-then-act race (two tasks starting close together can each see "0
  others running" before either registers) that also didn't apply at all to a direct call (e.g.
  `manage.py shell`, a management command) bypassing the Celery wrapper entirely. Confirmed via
  id-delta analysis: 2,681 of the 3,745 pairs had ids within delta <=5 of each other (682 adjacent,
  delta<=1) -- exactly the signature of two near-simultaneous processes each inserting the same 2
  faces back-to-back; the remaining ~1,064 pairs had much larger id deltas (up to 14,528), a
  separate, still-unidentified "reprocessed without clearing old faces" mechanism (see the TODO
  below).
  - **Fix, discussed and agreed with the user (who correctly flagged that a naive "mark done
    before actually done, then unset on failure" claim scheme is fragile)**: a new
    `common/advisory_lock.py` -- a Postgres advisory-lock context manager (`pg_try_advisory_lock`/
    `pg_advisory_unlock`, key = `zlib.crc32(name.encode())`), non-blocking, tied to the DB session
    rather than any row/file. Unlike a per-row "claim" flag, there's no separate unclaim-on-failure
    path to get wrong -- the lock is released automatically on any exit from the `with` block
    (including via exception) and, critically, also automatically by Postgres itself if the
    holding connection ever drops (crash, OOM-kill, container restart) -- no timeout/heartbeat
    logic needed. Scoped only to whatever name is passed in; doesn't touch any table, row, or other
    Postgres locking machinery, and has zero effect on any other lock name (verified: a lock on one
    name can be held while a different name is acquired freely, and it doesn't block ordinary
    queries at all -- only another `advisory_lock()` call using the *same* name).
  - **Applied to two tasks, per the user's explicit request to generalize beyond just this one
    fix**: `find_and_encode_faces()` (`face_manager.find_and_encode_faces` key) -- refactored into
    a thin `find_and_encode_faces()` wrapper that acquires the lock and skips (logging a warning)
    if already held, calling the unchanged original body (now `_find_and_encode_faces_locked()`)
    only once acquired; and `filepopulator/scripts.py`'s `add_from_root_dir()`
    (`filepopulator.add_from_root_dir` key), **replacing** the old `settings.LOCKFILE`
    file-based lock entirely -- that mechanism was a plain `os.path.isfile()` check-then-create
    with no wait/retry/timeout, and (per an already-known-but-unactioned TODO) a hard kill/OOM/
    container restart mid-run could leave the lockfile behind forever, silently no-op'ing every
    future scheduled run ("Locked!" then return) with no alerting -- the advisory lock closes that
    gap too, for free. `tasks.py`'s `process_faces()` had its now-redundant-and-racy `inspect().
    active()` check removed (the advisory lock inside `find_and_encode_faces()` supersedes it and
    is atomic). `settings.LOCKFILE` itself was left defined but is now dead, same as the
    already-unused `FACE_LOCKFILE`/`CLASSIFY_LOCKFILE` settings next to it -- not cleaned up.
  - Tested on `backend_upgrade`/`picasa_api_dev_test`: 6 new `common.tests.AdvisoryLockTests`
    (uncontended acquire, reacquire after release, release-on-exception, cross-session contention
    and release via a genuinely separate `psycopg2` connection -- Postgres advisory locks are
    reentrant *per session*, so simulating real contention requires a second connection, not
    another `with advisory_lock(...)` on the same one), 1 new
    `filepopulator.tests.ImageFileTests` case (`add_from_root_dir` does nothing at all while
    another session holds its lock, then works normally once free), and 1 new `face_manager.
    tests.FaceExtractorCorruptedImageTests` case (same shape, against real `find_and_encode_
    faces()` with real detection). Full fast suite: 290/292 passing (the same 2 pre-existing,
    unrelated failures as before -- fixture path availability and float32-precision rounding).
    Merged to `master` and deployed 2026-09-04 (`picasa_api` restarted to load the new code --
    no migration needed, no model/schema changes involved).
  - **RESOLVED (2026-09-04): root cause #2, the ~1,064 far-apart-id same-image duplicate pairs, is
    NOT a separate bug -- it's the same race condition at bulk-import scale.** The 98 distinct
    images involved don't scatter randomly across the library; they cluster into a handful of
    contiguous `ImageFile`-id ranges (sizes 23, 34, 68, 109), each corresponding to one bulk-import
    event -- e.g. the 109-image cluster is entirely `/photos/Completed/Pictures_finished/2018/
    Family Pictures/Erica Farewell/*`. That's the signature of a large batch of freshly-ingested
    photos landing in the DB with `isProcessed=False` all at once, then getting caught by the
    exact same race as root cause #1: two overlapping `find_and_encode_faces()` runs each
    independently working through the big unprocessed batch, colliding on many of its images. The
    larger id deltas simply reflect how many *other*, unrelated faces each run inserted before
    happening to reach the shared images -- not a different trigger. The advisory-lock fix above
    already fully covers this case too, regardless of batch size, since it prevents a second
    concurrent invocation from starting at all. No separate fix was needed.
  - **DONE (2026-09-04): `dedupe_overlapping_faces` management command, cleans up the duplicate
    rows already sitting in the database from both flavors above.** Finds groups of mutually-
    overlapping (IOU > 0.9, single-linkage via connected components) `Face` rows on the same
    image, collapses each group to one survivor, deletes the rest via real `Face.delete()` (not a
    bulk queryset delete, so thumbnail files get cleaned up too), and recomputes `Person.num_faces`/
    `num_possibilities`/`num_unverified_faces` for anyone affected. Survivor preference, in order:
    already-**validated** (never discard a completed human verification) > **has a real label**
    (`declared_name` isn't the blank sentinel -- added per the user's specific request, since a
    common real scenario is a human tagging one copy of a duplicate pair without knowing the other
    copy existed, leaving it blank forever) > has **kps** populated (lets a later reencode
    reproduce the exact embedding without re-detecting) > lowest id as a final deterministic
    tiebreaker. `--dry-run`/`--yes` flags match this repo's established cleanup-command
    convention. 12 tests covering the connected-components grouping, each survivor-preference
    criterion individually and their priority ordering, re-run-finds-nothing idempotency, and
    thumbnail-file cleanup. Dry-run against real production data (no upper cap on faces-per-image,
    unlike the earlier ad hoc investigation script which capped at 8 and undercounted): 8,264
    duplicate groups, 8,368 Face rows to delete. **Run for real against production 2026-09-04:
    8,368 duplicate faces deleted, face counts recomputed for 227 affected people, ~1 minute.
    Confirmed clean afterward -- a second `--dry-run` immediately found 0 groups.**
- **DONE (2026-09-04): the duplicate-ImageFile bug above, root-caused and fixed.** Confirmed via
  the user's own example (faces 1060610/1076848, on `ImageFile`s 340565 and 346648): same
  `pixel_hash` MD5 (`c3e6fce0cc3c8adeb545380d44acc826`), same dimensions, same `dateTaken`,
  different paths -- already flagged in `SimilarImagePair` (`hamming_distance=0`) since 2026-08-27,
  and (crucially) already correctly recorded as a `DuplicateFile` too. **Root cause**:
  `create_image_file()`'s "pixel_hash matches an existing, still-present file" branch
  (`filepopulator/scripts.py`) correctly created the `DuplicateFile` record but was **missing a
  `return` statement**, so execution fell through to the bottom of the function and created a full
  `ImageFile` row for it anyway, every single time -- every real duplicate got BOTH correctly
  flagged AND incorrectly given its own row. The same gap existed in the sibling `len(exist_with_
  same_hash) > 1` branch (multiple existing rows already share the hash) when none of them had a
  missing file to "move into." Fixed by adding the missing `return`s in both branches.
  **Secondary fix, per the user's explicit choice when asked**: rather than trusting a bare
  `pixel_hash` MD5 match as sufficient proof of duplicate content, added a real pixel-level
  verification (`_pixel_arrays_match()`) -- re-decodes the candidate file and compares actual
  pixel arrays before trusting the hash, so a (deliberately synthetic, essentially impossible for
  real photos) MD5 collision between two genuinely different images still correctly creates its
  own row rather than being wrongly discarded as a duplicate. This preserves an existing test
  (`test_same_pixel_hash`) that constructs exactly such a collision on purpose. Three other
  existing tests (`test_same_picture_two_paths`, `test_image_path_changes_two_instances`,
  `test_move_id_stay_same`) had explicitly asserted the OLD (buggy) two-rows-for-one-photo
  behavior as their documented "expected outcome" -- rewritten to assert the corrected behavior
  instead, plus `test_bulk_add`'s blanket "every fixture file becomes its own ImageFile row"
  assertion loosened to "becomes an ImageFile OR a recorded DuplicateFile," since the real
  validation fixture directory has always contained a deliberate duplicate pair. Full fast suite:
  305/307 passing (same 2 pre-existing, unrelated failures as always).
  - **Existing contamination, mocked up (dry-run), not yet cleaned up: `filepopulator/management/
    commands/merge_duplicate_imagefiles.py`.** Fixing the bug going forward doesn't undo the
    1,197 already-contaminated `ImageFile` rows sitting in production (an `ImageFile` whose own
    filename also has a `DuplicateFile` record). Real risk found before building anything: **5,009
    faces sit on those 1,197 rows, 1,190 of them validated, 4,190 carrying a real label** -- a
    human tagged these without knowing the photo was a duplicate, so a naive "just delete the
    contaminated rows" cleanup would have destroyed real completed work. Per the user's explicit
    call ("we should keep all the info we have"), the command instead: finds each contaminated
    row's "primary" (another `ImageFile` sharing the same `pixel_hash`, itself not flagged as a
    duplicate), reassigns every `Face` on the duplicate over to the primary
    (`Face.objects.filter(...).update(source_image_file=primary)` -- bulk, valid since the two
    rows are pixel-identical and share width/height), then **collapses any resulting same-image
    duplicate face pairs on the primary** by reusing `dedupe_overlapping_faces`'s own grouping/
    survivor-preference logic directly (imported, not reimplemented) -- since the two source
    images are pixel-identical, a face present on both will usually land at the same box after the
    transfer, needing the same validated > labeled > kps > lowest-id collapse. Finally deletes the
    now-empty duplicate `ImageFile` row (`ImageFile.delete()` already cleans up its own thumbnail
    files and any remaining faces properly). A contaminated row whose primary can't be found
    (already separately removed, etc.) is deliberately left alone rather than guessed at. 6 new
    tests (transfer, collapse-prefers-validated, unresolved-left-alone, dry-run-no-op, person-count
    recompute on a losing collapse, re-run-finds-nothing), full fast suite still green (311/313,
    same 2 pre-existing failures). **Dry-run against real production data: 1,197 contaminated rows
    -- 948 resolvable (a clear primary found), 249 unresolved (no primary found, left alone), 4,202
    faces would be transferred.** **Run for real 2026-09-04: 948 merged and deleted, 4,202 faces
    transferred, 4,096 of those collapsed as exact-duplicate pairs on the primary (nearly all --
    expected, since the two source photos were pixel-identical), ~106 survived as genuinely new
    faces only one of the two copies had detected, 48 people's face counts recomputed. App healthy
    afterward; remaining contaminated count matched the 249 unresolved exactly.**
    Investigated the 249 unresolved further at the user's request: **247 of them have no other
    ImageFile row sharing their content at all** -- their primary was deleted at some point (e.g.
    `delete_removed_photos()` ran after the original file vanished from disk), leaving the
    duplicate-flagged copy stranded with nothing to merge into (but nothing lost either -- there
    was only ever the one surviving copy). **The other 2 are a single reciprocal-flagging pair**
    (`.../Kings Island/20251025_194106.jpg` and `.../cbbbc672-...-copied-media~2.jpg`) -- both real
    photos exist, but each independently ended up with its own `DuplicateFile` record pointing at
    the other, so the command's safety check (only pick a primary that isn't itself flagged)
    excludes both. Not fixed -- a narrow, 2-row edge case, noted here rather than chased further.
  - **DONE (2026-09-04): `DuplicateFile.original` FK, closing the root design gap the 247-of-249
    case above exposed.** `DuplicateFile` previously stored only a bare `filename` -- no reference
    at all to which primary `ImageFile` it was a duplicate of, so there was no way to notice "the
    primary this pointed at just got deleted." Added `original = ForeignKey(ImageFile,
    on_delete=CASCADE, null=True)` (migration `filepopulator.0006_duplicatefile_original`),
    populated by `create_image_file()`'s two duplicate-recording branches going forward. The
    `CASCADE` is the actual point: when a primary `ImageFile` is later deleted, its `DuplicateFile`
    records now go with it automatically, freeing the surviving duplicate file to be genuinely
    re-ingested as a real photo the next time it's scanned -- exactly the gap that stranded the
    247. (The reverse direction -- the *duplicate* file being deleted instead -- needs no special
    handling: the primary stays completely valid either way, and a `DuplicateFile` row whose own
    path no longer exists is harmless, inert clutter, not a correctness problem.)
    **`backfill_duplicatefile_original`**: one-time (safely re-runnable) command resolving the
    ~17,330 pre-existing `DuplicateFile` rows that predate this field -- for each with `original`
    still NULL: if its own file is gone, delete the row (moot either way); if the file decodes and
    exactly one `ImageFile` shares its `pixel_hash`, set `original`; if none do, delete the row
    (same reasoning as the 247 case -- frees the sole surviving copy); if the file exists but fails
    to decode, or more than one current `ImageFile` shares the hash (ambiguous), leave it alone
    rather than guess. 12 new tests total (FK population on both create_image_file() branches,
    CASCADE-on-primary-delete, and the backfill's four outcomes -- resolved / file-gone-deleted /
    no-primary-deleted / corrupted-left-alone -- plus dry-run and re-run idempotency). Full fast
    suite: 319/321 passing (same 2 pre-existing, unrelated failures).

**DONE (2026-09-04): cut `faceAssigner`'s daily encoding cache from 7.08GB to 2.80GB resident
memory and 13.7s to 0.84s load time (both measured against the real production cache).** Started
from the user asking whether the day-scoped encoding cache (`load_encodings()`,
`/models/face_assign_preload.pkl`) could be made to persist in memory across runs, and how much
RAM that would take -- "just a couple gigs?" Investigated empirically rather than guessing:
production's actual cache file was 2.4GB on disk; loading it took 13.71s and left the process
holding **7.08GB resident RAM** (`VmRSS`, confirmed real and not reclaimable transient overhead
via `gc.collect()`/`malloc_trim` -- neither changed it). That's well more than "a couple gigs,"
and the gap turned out to be a genuine bug, not requirements: `candidate_dict` (built inside
`load_encodings()`) stored every face's 512-d embedding a *second* time -- as a Python `list` of
512 individually-boxed floats inside a pandas object-dtype DataFrame column -- duplicating the
exact same data already held compactly in `embedding_dict` (a packed numpy array). A full-codebase
grep confirmed `candidate_dict`'s own embedding column is never read by anything except the very
next few lines of `load_encodings()` itself, which builds `embedding_dict`/`norm_dict` from it and
then never touches it again -- pure redundant storage, just in a ~8x-more-expensive form (a Python
list of boxed floats vs. a packed array). Verified this was the actual cause, not a red herring,
before touching code: dropping that one column from the loaded pickle and re-measuring showed
current `VmRSS` fall from 7.08GB to 2.80GB immediately. Fixed by never storing that column in the
first place -- `load_encodings()` now drops it from each person's cached DataFrame right after
using it to build `embedding_dict` (`face_manager/assign_faces.py`). Re-measured against a
from-scratch save/load of the corrected structure: pickle file 2.4GB -> 1.19GB, load time 13.7s ->
**0.84s (a real ~16x speedup)**, resident memory 7.08GB -> 2.80GB -- matching the user's original
"couple gigs" estimate once the redundant copy was gone. No functional change: the one existing
test asserting `len(candidate_dict[person_id])` still passes (row count is unaffected by dropping
a column). Full fast suite: 317/317 passing.
- **Separately, also fixed while investigating the batch-size question that prompted this**:
  `faceAssigner.execute()`'s old <=100-unassigned-faces bug turned out to already be fixed (see
  the corrected TODO entry above) -- confirmed no batch-size gate remains anywhere in the file.
- **Genuine in-memory persistence across Celery task runs (not just a faster per-run disk
  reload) was discussed but not built.** `picasa_api`'s Celery workers run with
  `--max-tasks-per-child 3` -- each worker process is killed and replaced after 3 tasks, so a
  plain module-level Python cache would only survive ~3 task runs regardless, not reliably "until
  tomorrow." Real persistence across many runs would need a dedicated worker/queue for
  `face_manager.assign_faces` with `--max-tasks-per-child` effectively unlimited, holding the
  cache as a module-level global (checked against the same day/signature invalidation
  `load_encodings()` already uses) -- a bigger, deliberately-deferred change (new queue routing, a
  dedicated worker process, reasoning about what happens if that worker crashes mid-day). Given
  the disk-reload path now costs well under a second, the user opted for the smaller fix above
  instead of taking this on.
- **DONE (2026-09-04): relaxed the cache-staleness window from 1 day to 3
  (`faceAssigner.CACHE_MAX_AGE_DAYS`)**, per the user's follow-up question about SSD wear.
  Confirmed reads cost essentially nothing on SSD wear-wise (wear comes from program/erase
  cycles, i.e. writes; reads only cause "read disturb," a well-managed background concern
  handled transparently by drive firmware) -- so the real tradeoff `CACHE_MAX_AGE_DAYS` controls
  is write frequency (how often the ~1.1GB cache file gets rewritten), not read cost. A brand-new
  qualifying person (freshly crossing `MIN_NUM_FACES`) is still picked up immediately regardless
  of this window -- the per-call top-up loop in `load_encodings()` already ran unconditionally on
  every call before this change and still does, so this didn't need any new logic, just
  confirming the existing behavior covered it. 2 existing tests (`test_next_day_no_changes_keeps_
  cache`/`test_next_day_with_changes_rebuilds_cache`) hardcoded a 1-day offset to simulate
  staleness and would have silently stopped exercising the stale path at all under the new
  3-day window -- renamed and parameterized against `CACHE_MAX_AGE_DAYS` instead of a literal
  `timedelta(days=1)`, plus a new test added for the "still within the (now 3-day) window"
  case explicitly. Full fast suite: 318/318 passing.

**Face-classification outlier-rejection: investigation and ideas (2026-08-27).** Started from a
real user observation: `face_manager/assign_faces.py`'s `classify_unassigned()` is good at
correctly matching faces to known people, but frequently proposes outlier faces as matches too.
Investigated via a real experiment methodology -- `.ignore`/`.realignore` confirmed faces used as
known-outlier ("negative") queries, leave-one-out on real confirmed faces as "positive" queries,
scored against the actual 441-person/273k-face gallery `assign_faces.py` uses. Findings and open
ideas, so a future session doesn't have to redo this from scratch:
- **IMPLEMENTED, DEPLOYED, AND REPROCESSED (2026-08-27/28): gallery-size-adaptive `p99` gate, 4
  buckets.**
  `classify_unassigned()` currently gates accept/reject on `sim_max` (line ~300-312) -- a pure
  1-nearest-neighbor comparison, maximally vulnerable to any single noisy/mislabeled face in a
  person's gallery. The code already computes `sim_99th` per candidate but only uses it as the
  display "weight," never as the actual gate. A single global swap from `sim_max` to `sim_99th`
  helps FPR but costs TPR unevenly -- a much bigger, size-dependent re-run (up to 50 leave-one-out
  holdouts per person, 3000 negatives, `p99` computed via one batched `np.percentile` call per
  candidate rather than N separate calls) found the TPR cost is concentrated almost entirely on
  large-gallery people (TPR 91.5%->60.9% for 1000+-face people at `ASSIGN_THRESH=0.6`, vs
  85.2%->84.4% for 10-25-face people, essentially free there). Root cause: percentile rank scales
  with sample size, so a fixed percentile is silently stricter for people with more faces --
  exactly backwards from what's wanted, since large/growing galleries are the ones future photos
  keep landing in. **Final design**: split the 441-person gallery into 4 buckets by face count --
  `[10,50)`, `[50,200)`, `[200,500)`, `[500+)` -- each with its own `p99` threshold calibrated to
  its own target TPR, searching *within* that bucket only (not the full mixed population, which
  matters -- see below). Chosen thresholds/targets, from the fully cached experiment data:
  `[10,50)` thresh=0.558 (target 90% TPR), `[50,200)` thresh=0.551 (90%), `[200,500)` thresh=0.486
  (90%), `[500+)` thresh=0.394 (target bumped to **95%** specifically for this bucket, per user
  request to further protect TPR for the most-photographed/fastest-growing people -- costs the
  system ~0.4pp of total FPR, deliberately accepted). **Critical methodological correction made
  mid-investigation**: an early "blended FPR" metric (positive-count-weighted average of each
  bucket's own marginal FPR) *looked* like bucketing cut total FPR to a third (2.97%->1.00% for a
  3-bucket version) -- this was wrong. A real unassigned face gets checked against all buckets at
  once (identity unknown in advance), so the metric that matters is the *joint/union* FPR
  (fraction of negatives accepted by *any* bucket), which came out at 2.9-3.3% -- essentially
  identical to the original single-threshold `p99` approach. **Bucketing's real, validated value
  is TPR fairness across gallery sizes at roughly the same total FPR cost, not a lower total FPR**
  -- confirmed by sweeping the joint FPR across TPR targets 70-95%, where bucketed and unbucketed
  track each other almost exactly at every point. Also confirmed (properly, holding TPR fixed
  this time, unlike an initial flawed attempt that let TPR collapse to "prove" a false win): adding
  `p50` as a second AND-condition alongside `p99` does not meaningfully help in any bucket
  (largest observed gain: 1.03%->0.97% FPR at matched 90% TPR, well within noise for a 3000-sample
  test) -- consistent with the logistic-regression finding below that percentiles of the same
  distribution are too correlated to combine for real gain. **Implemented and live in
  production.** The full 140,479-face unassigned-face library was reprocessed overnight
  (`faceAssigner().execute(redo_all=True)`, ~8 hours, single-threaded -- see the reverted
  multi-threading note below): 16,547 faces (11.8%) got a confident real-person suggestion,
  123,933 (88.2%) fell back to `.ignore` -- the ignore-heavy split is expected and intentional,
  trading auto-match volume for a much lower false-positive rate, exactly as designed. Only 1
  face failed (thumbnail file missing from disk, unrelated to this change -- confirmed isolated
  via a random sample, not systemic; fixed by deleting it and its sibling face on the same image,
  one of them a previously-confirmed "Gwendolyn Lewis" tag now needing re-confirmation, and
  marking the image `isProcessed=False` for redetection). **User's own anecdotal read after the
  reprocess: "the face classification is looking a LOT better."**
  - **Speedup work done alongside the reprocess, all kept except threading**: batched
    `Face.save()` calls (`Face.set_possible_person()` gained a `save=False` option --
    `classify_unassigned()` could call it up to 5x per face, each a real ~20ms validated save);
    removed a fully-dead per-face `source_image_file.dateTaken` fetch (computed for a
    commented-out debug print, never otherwise used, but still triggered a real query); added
    `select_related('declared_name')` to `execute()`'s queryset (checked on every face, wasn't
    prefetched); vectorized the per-candidate comparison into one big matmul against a
    concatenated gallery matrix instead of ~441 separate small ones
    (`_build_concatenated_gallery()`); and cached `Person` objects (`_build_person_cache()`) so
    `set_possible_person()` skips its own `Person.objects.get()` round trip. **Multi-threading
    (`num_threads` param, `ThreadPoolExecutor`) was tried and reverted** -- measured against the
    real reprocess, 6 threads gave no real speedup over the single-threaded-but-optimized version
    (~5 it/s either way) despite high CPU usage, most likely because numpy's matmul already uses
    multiple BLAS threads per call, so N Python threads oversubscribe the same cores rather than
    dividing work. Not worth the added complexity (thread-local DB connection handling,
    `TransactionTestCase`-only test coverage) for zero measured benefit.
  - **Real bug caught and fixed mid-implementation, not by a test**: the "trueing up" pass
    (`Person.objects.all()`, recomputing `num_faces`/`num_possibilities`/`num_unverified_faces`)
    briefly ended up inside the per-face helper during the threading work instead of staying in
    `execute()` -- ran once per face instead of once per `execute()` call, turning the reprocess's
    ETA from ~10 hours into ~92 projected. Caught by watching the real run's rate, not by the test
    suite (existing coverage only checked final counts, not call counts) -- a regression test
    (`ExecuteTrueingUpTests`) was added afterward and is kept even post-threading-revert.
- **Tested and rejected: combining multiple percentiles (p50/p75/p90/p95/p99/max) via logistic
  regression.** AUC barely improved over `p99` alone (0.968 vs 0.967) -- percentiles of the same
  similarity distribution are too correlated with each other to add real complementary signal.
  Top-k averaging (top-10/top-25) was *worse* than plain percentiles, likely because a fixed-k
  average dilutes badly for people whose galleries are barely above `MIN_NUM_FACES=10`, while a
  percentile automatically scales with each person's own gallery size.
- **Validated: same-person cosine similarity decays substantially with photo date gap** (0.544
  mean similarity at 0-3mo gap vs 0.285 at 15+yr gap, correlation -0.37, 3.4M same-person pairs
  across 318 people). Real, strong effect -- but:
- **Tested and rejected (for accept/reject purposes): general date-windowing** (only compare
  against a candidate's own faces within N years of the query, N in 1-10). TPR barely moved,
  FPR didn't improve. Explained by a follow-up measurement: windowing does cut the average
  candidate field size a lot (e.g. only ~41% of 440 people survive a 1-year window) but the
  people most likely to cause false positives -- those with large, temporally-broad galleries --
  survive *any* window width, so windowing filters out people who were never going to win
  anyway, not the actual troublemakers. Caveat: tight windows leave few points per candidate, so
  the windowed metric falls back to `max` (noisier), which may itself be part of why it didn't
  help -- not fully isolated from the small-sample-instability confound.
- **Tested and blocked by a separate data-quality problem: birth-year-based hard cutoff.**
  `Face.detected_age` (insightface, populated on 631,874/637,960 faces already) can estimate a
  person's birth year as `median(photo_year - detected_age)` across their gallery -- precise to
  well under a year for people with 1000+ faces (bootstrap 95% CI ~0.9yr), but investigation
  found the *aggregate* precision doesn't mean the estimate is *accurate*: a known-preschooler
  ("Liam Lewis") had 99.6% of his 17,485 confirmed faces show `detected_age > 15` (median 46) --
  and two other people with very different presumed true ages (Nathaniel, Benjamin) showed
  nearly identical medians (42, 46). That consistency suggests `detected_age` may not be a
  usable per-photo age signal for this pipeline at all (possibly landing in a narrow band
  regardless of true age -- face-crop quality/resolution feeding the age model, or a bug in how
  the value is read from insightface's output, not investigated further). **Tabled by the user
  pending a visual sanity-check** (pull a handful of real face thumbnails next to their
  `detected_age` and eyeball whether it's remotely plausible) before reviving this idea.
- **Separately discovered, real, previously-undocumented bug: `Face.dateTakenUTC` corruption.**
  ~7,177+ faces have wildly corrupted dates (one as far back as year 0102 AD), and there are
  large clusters of many *distinct* source images sharing one identical to-the-second timestamp
  (e.g. 415 distinct images all at `2000-10-20 12:06:30`, all `.realignore` faces) -- not
  plausible for real photography, smells like a fallback/default-date bug rather than genuine
  EXIF data. Not investigated further this session (worked around via a sane `[1990, 2027]`
  bound for the birth-year experiments); worth a real look given it could affect other
  date-dependent logic (`Directory.average_date_taken()`, the geocode/date-decay work, etc.).
- **Per-person calibrated threshold: superseded by the 4-bucket design above**, which is a
  coarser (gallery-size-based, not fully per-person) version of the same idea and is now
  data-validated and spec'd -- no need to separately pursue a per-person version unless the
  4-bucket design proves insufficient in practice.
- **Tested and rejected: covariance-aware (Mahalanobis) distance.** Tried a diagonal-only
  approximation (per-dimension variance, not a full 512x512 covariance -- see the practical
  blocker below) as a feature alongside `max`/percentiles/gap-features in a logistic regression,
  restricted to large-gallery (200+ faces) people where there's enough data to estimate even the
  diagonal reliably. Single-feature AUC=0.976, *worse* than `p99` alone (0.988), and it got a
  near-zero, wrong-signed coefficient in the combined model -- didn't pull its weight. A full
  (non-diagonal) Mahalanobis distance was never tried -- would need shrinkage estimation (e.g.
  Ledoit-Wolf) or PCA dimensionality reduction to be estimable at all given most of this gallery's
  people have far fewer faces than the 512 embedding dimensions -- but given the diagonal version
  already underperformed, a full version isn't an obvious next step without a reason to expect
  the *correlations* between dimensions (the part diagonal ignores) to carry the missing signal.
  **Dropped from the TODO list (2026-09-04) per the user's call**: a full 512x512 covariance
  would be underspecified for most people's actual gallery sizes anyway (needing shrinkage/PCA
  just to be estimable at all, per above), and the already-tried diagonal approximation gave no
  reason to expect the full version would help -- not worth pursuing.
- **Brainstormed, not yet tried**:
  - **Co-occurrence / social-context prior** -- if other faces in the *same photo* are already
    confidently identified, and those people are frequently photographed together with a given
    candidate (siblings, spouse, etc.), that's a real prior signal independent of the embedding
    entirely (how early Picasa/Google Photos boosted tagging accuracy). Bigger lift -- needs new
    co-occurrence-statistics infrastructure, not scoped.
  - **Directory/event context prior** -- faces from the same source folder or day tend to
    recur; if a directory is already heavily populated with a specific family group's confirmed
    faces, that shifts the prior for an unlabeled face in that same directory. Also not scoped.
  - **Cluster-then-recover the `.ignore` bucket -- investigated 2026-09-03, real progress, still
    open.** Started from the idea above (cluster within `.ignore`/suggested faces to recover TPR
    lost to the conservative gate) but evolved once real experiments started: the actual driving
    goal became "reduce human review cognitive load" more than "recover matches automatically" --
    grouping visually-similar faces so a person can spot-check a whole run at once, not
    necessarily reconstructing per-person identity.
    - **Methods tried against the ~100,405-face `poss_ident1=.ignore` (suggested, not confirmed)
      population**, sampled at n=10k-40k throughout: HDBSCAN (`eom` selection kept collapsing into
      one dominant blob covering up to ~40% of the sample the moment `min_cluster_size>=3`; `leaf`
      selection avoided the blob but fragmented into tiny 3-15-face pieces with 90%+ noise;
      `max_cluster_size` capping HDBSCAN's `eom` output revealed a real structural gap -- capped
      output plateaus identically across a wide range of cap values, e.g. Erica's own gallery
      showed *zero* change from cap=100 to cap=200 -- there's no smooth continuum of medium
      clusters hiding inside the blob, just "small pieces" or "the whole blob," nothing between).
      kNN-graph + Louvain community detection and kNN + plain connected-components were also
      tried and both reproduced the same one-dominant-blob failure mode (classic single-linkage
      chaining: A-B-C-D all merge transitively even if A and D aren't alike). DBSCAN with epsilon
      derived from the already-calibrated `classify_unassigned()` cosine thresholds (0.4–0.6) did
      the same. **Complete-linkage agglomerative clustering (`sklearn.cluster.
      AgglomerativeClustering(linkage='complete', distance_threshold=..., metric='euclidean')` on
      L2-normalized embeddings) was the one method that never produced a giant blob**, at any
      threshold or scale tested (confirmed at both a single person's ~15k-face gallery and 10k/20k
      chunks of the real heterogeneous `.ignore` population) -- because it requires the *worst*
      pairwise distance within a candidate cluster to still be under threshold, not just one
      bridging pair, which structurally blocks the chaining every single-linkage-family method
      (DBSCAN, connected-components, Louvain, HDBSCAN's own mutual-reachability core) suffered
      from. `average` linkage sits in between -- less bloblike than single-linkage, but still
      blobbed badly at large n (Erica: max cluster 8222 at cos=0.4, vs. complete linkage's max
      500 at the same threshold on the same data).
    - **Verdict on the `.ignore` population specifically: abandoned.** Chunking 100k into 10k or
      20k pieces (accepting some cross-chunk matches would be missed) and running complete linkage
      at cos_threshold=0.5 found real structure (10k chunks: 4,519 groups, 24.6% of the population
      grouped, ~5 min total; 20k chunks: 4,940 groups, 27.3% grouped, ~11 min -- diminishing
      returns on chunk size, same shape as everything else in this investigation) -- but visually
      inspecting real contact sheets of the resulting groups (built from real production face
      thumbnails, both at cos=0.5 and a stricter cos=0.7) showed the groups were NOT
      single-identity -- e.g. the largest 0.5-threshold group (36 faces) was multiple different
      (related) family members mixed together, not one person, even at 0.7. The user's own
      conclusion: not worth pursuing further for this population -- these embeddings are simply
      too unreliable/low-information (matches this population's already-known skew toward
      small/blurry/low-quality detections) for similarity-based grouping to reliably separate
      individuals, regardless of algorithm or threshold.
    - **Pivoted to confirmed people's own UNVERIFIED faces instead -- this direction looks
      genuinely promising and is what's being built now.** Tested complete linkage (thresholds
      0.65/0.7/0.75) on real confirmed galleries spanning size buckets: Mack Holyoak (14),
      Cutler Kid (35), Elder Thomas (63), Benjamin Stevens (106), Angie (242), Alissa Lewis (963),
      Peter Van Katwyk (1818), Erica Bradshaw (14,983). At cos=0.7, contact sheets of Erica's
      5 biggest resulting sub-clusters (sizes 99, 84, 59, 43, 38) showed 4 of the 5 genuinely
      visually coherent (real, consistent-looking sub-groups/eras of the same person) -- **the
      user's own read: "pretty coherent."** The one exception (the 99-face group) looked
      scattershot despite being the most stable/reproducible cluster boundary across every method
      tried (HDBSCAN uncapped, the `max_cluster_size` sweep, and complete linkage all
      independently drew the same line around it) -- investigating why led to a real, separate bug
      fix (see below): those 99 "faces" were real, distinct photos that all happened to carry the
      literal same placeholder sentinel embedding, not genuine visual similarity.
    - **Real bugs found and FIXED via this investigation, both already merged to
      `master`/`backend_upgrade` and deployed:**
      1. **`reencode_missing_faces()` only matched NULL `face_encoding_512`, missing a second,
         larger population of broken faces.** `update_list_of_no_matching_detects()`
         (`face_extract_encode.py`) stamps `settings.NON_DETECTED_FACE_ENCODING` (`[-999]*512`)
         onto a face whose box wasn't matched to any detection during a full-image reprocessing
         pass -- a real, declared-to-a-real-person face left with a garbage embedding, not NULL,
         so invisible to the original filter and not `.ignore`/`.realignore` either. Found: 1,207
         faces database-wide had this sentinel; the existing `cleanup_chronically_unmatched_faces`
         command (scoped to a narrower, already-fixed orientation-6/8 bug) only caught 1 of them.
         Fixed by extending `reencode_missing_faces()`'s selection query to also match the
         sentinel value directly (`Q(face_encoding_512__isnull=True) |
         Q(face_encoding_512=settings.NON_DETECTED_FACE_ENCODING)`), treating it the same as NULL.
      2. **File-descriptor leak in `common/open_img_oriented.py`, found while actually running the
         fix above against the real 1,219 affected faces.** The run hit `[Errno 24] Too many open
         files` partway through (`ulimit -n` 1024), making 206 perfectly good images fail with a
         misleading "decode error" -- not corruption, resource exhaustion. Root cause: Pillow's
         `Image.load()` is documented to close the underlying file once decoded, but neither
         `_getexif()` (metadata-only read) nor `convert()`/`transpose()` (each returns an
         independent new object) guaranteed that ever happened for the file object
         `PIL.Image.open()` originally returned -- a tight loop over many images (exactly what
         `reencode_missing_faces()` and `find_and_encode_faces()` both do) leaked one fd per call.
         Fixed by keeping a reference to the originally-opened image and calling `.load()` on it
         in a `finally` block regardless of which code path ran -- harmless no-op if already
         loaded some other way, never touches whatever derived object is actually returned.
      Both fixes covered by regression tests (`ReencodeMissingFacesTests`, `OpenImgOrientedTests`)
      and confirmed against real production data: re-ran `reencode_missing_faces()` against
      production after the sentinel-query fix landed (1,219 eligible faces, ~22 min), hit the fd
      leak partway through (206 failures), fixed the leak, re-ran against just the 206 remaining
      -- see whether that final re-run's result got recorded below if this note wasn't updated
      again afterward.
    - **DONE (2026-09-03/04): `Face.verification_cluster_group` nightly clustering feature,
      built as planned, merged to `master` and deployed.** New
      nullable `IntegerField` on `Face` (migration `0008_face_verification_cluster_group`),
      populated by `face_manager/verification_clustering.py`'s `cluster_all_unverified_faces()`
      -- complete-linkage clustering (`sklearn.cluster.AgglomerativeClustering(linkage='complete',
      metric='euclidean')` on L2-normalized embeddings, cos threshold **0.6 default** (changed
      from the original 0.7 via 0.65 -- see the dated note below), configurable via a function
      argument, `dist = sqrt(2 - 2*cos_sim)`) run independently **per real person**
      (never mixing galleries, one `AgglomerativeClustering` call per person) over
      `eligible_faces_queryset()`: **unverified** (`validated=False`), **valid-encoding**
      (excludes NULL and the `NON_DETECTED_FACE_ENCODING` sentinel), **non-ignore**
      (`declared_name__person_name` not in `settings.IGNORED_NAMES`) faces. Group ids are
      0-indexed per person (independent across people, no global uniqueness) and assigned only to
      clusters of size >=2; singletons and ineligible faces are left/reset to `NULL`. Wired into a
      new `face_manager.cluster_unverified_faces` Celery task (`face_manager/tasks.py`, same
      already-running-lock-check pattern as the other scheduled tasks here), scheduled nightly at
      1am Eastern (`CELERY_BEAT_SCHEDULE['cluster_unverified_faces']`, `picasa/settings.py`) --
      deliberately ahead of `db_picasa`'s 2am daily backup and 3am-Monday vacuum-swap jobs so
      nothing overlaps. Each run clears **every** `Face.verification_cluster_group` value db-wide
      first, then rebuilds from scratch (no attempt to preserve group identity night-to-night, per
      the original plan). **The "clear immediately on reassignment" requirement is implemented in
      three places, not the originally-planned two**: `Face.associate_person()` and
      `Face.verify_person_in_image()` as planned, plus `Face.reset_to_pool()` -- found while
      implementing that `reset_to_pool()` is a third, actively-used (`api/mobile_views.py`)
      assignment-changing path distinct from `associate_person()`, so it needed the same hook to
      actually satisfy the "any time a face's assignment changes for any reason" requirement.
      Tested via 9 new cases in `face_manager/tests.py::VerificationClusterGroupTests` (synthetic
      512-d embeddings clustered around near-orthogonal base directions, not real face data):
      distinct-cluster-ids-plus-singleton-stays-null, per-person id independence, and every
      exclusion (`validated=True`, ignored sentinel names, NULL encoding, sentinel encoding) each
      checked individually, a nightly-clears-stale-groups case, and one hook test per
      reassignment path. Full fast suite run afterward: 279/281 passing, the 2 failures pre-existing
      and unrelated (a corrupted-image fixture path not mounted in that exec context, and an
      already-known float32-precision rounding assertion) -- confirmed neither touches this
      feature's files. Deployed to production (`face_manager.0008` migrated, `picasa_api`
      restarted -- confirmed `face_manager.cluster_unverified_faces` registered in `celery
      inspect registered` afterward) and run once immediately as a manual backfill
      (`cluster_all_unverified_faces()` via `manage.py shell`) rather than waiting for the first
      1am scheduled run. **Real backfill result (2026-09-04): 30 people clustered, 37,812 of
      65,383 eligible faces (57.8%) grouped, ~2m36s wall time** -- well under the ~10-minute
      estimate from the original 65,371-face scoping count (49 people at scoping time vs 30
      actually producing a real group here; the rest were either singletons or already covered
      by `MIN_NUM_FACES`-style small-gallery exclusions upstream, not investigated further).
      Largest galleries grouped: Nathaniel Lewis (6,578 grouped faces), Liam Lewis (5,689),
      Jessica Lewis (2,955), Gwendolyn Lewis (2,633), Benjamin Lewis (1,850). The frontend surface
      for actually using this (grouped review UI) remains explicitly out of scope for this repo --
      the user plans to design that separately once the backend/data side is live.
      **Real backup-restore rehearsal, same day (2026-09-04)**: at the user's request, restored
      that morning's `picasa_db_2026-09-04.tar.zst` (02:03am backup, predating both the migration
      and the backfill above) into a scratch DB, verified row counts matched exactly
      (638,116 faces / 206,666 images), then promoted it to replace live `picasa` (old DB kept
      aside as `picasa_prerestore_2026_09_04`, `picasa_api` stopped/restarted around the swap,
      same mechanism as `weekly_vacuum_swap.py`), re-applied migration `0008`, and re-ran the
      clustering -- got the identical 30-people/37,812-faces result, confirming the whole
      pipeline (migrate + cluster) works cleanly against a real restored backup, not just the
      already-live DB. The three parked DB generations (`picasa_prerestore_2026_09_04`,
      `picasa_prevacuum_2026_08_31`, `picasa_pre_reset_2026_08_26`) were dropped afterward at the
      user's request, once satisfied with the result -- only `picasa` remains.
      **Threshold walked down 0.7 -> 0.65 -> 0.6, same day (2026-09-04)**: the user asked to see
      each looser threshold's real effect in turn (the 0.65/0.7 comparison had been tested during
      the original investigation, but only via cached experiment data, never as a real production
      run). Each step re-ran `cluster_all_unverified_faces(cos_threshold=...)` against live
      production -- same 30 people every time, coverage of the 65,383 eligible faces climbing
      each step: 37,812 (57.8%) at 0.7, 41,756 (63.9%) at 0.65, 46,236 (70.7%) at 0.6. At 0.6 the
      user visually spot-checked real groups in the frontend (not just the raw counts) and called
      it good -- kept. `DEFAULT_COS_THRESHOLD` in `verification_clustering.py` is now `0.6`, so
      the nightly 1am task uses it going forward too, not just these one-off runs. No contact-sheet
      audit at 0.6 was done outside the frontend spot-check -- worth remembering if quality
      complaints ever come in, since the original investigation's own visual-coherence checks
      (at 0.7, on Erica's gallery) don't automatically extend to a looser threshold.
  - **Looser branch for faces with `.ignore` already in their reject list (2026-08-28).** A face
    whose `rejected_fields` contains `.ignore` means a human previously declined an *auto-
    proposed* soft-ignore for it -- i.e. someone already looked and said "no, this is a real
    person, not noise." The user's hypothesis: these faces are disproportionately likely to
    belong to one of the large-gallery people (the ones with enough photos that `.another_ignore`/
    `.ignore` kept getting auto-suggested for their harder shots), so it's worth giving this
    specific subset its own classification pass with looser (lower) `p99` thresholds than the
    standard 4-bucket gate, rather than only ever comparing them at the same conservative
    operating point as a brand-new unlabeled face. Complements the "cluster-then-recover the
    `.ignore` bucket" idea just above but is narrower/cheaper to try first: no new clustering
    infra needed, just a query for `rejected_fields` containing `.ignore` plus a second gate
    threshold (or bucket set) applied only to that subset. Not scoped or started -- would want to
    validate empirically first (same TPR/FPR methodology as the original bucket-threshold work)
    that this subset's true-positive rate at a loosened threshold is actually higher than a
    random unassigned face's, before shipping a change that's easy to mis-tune.
- Experiment scripts and cached data (gallery embeddings, negative pool, full per-query/per-
  person/per-percentile profiles, all keyed by person/face id with dates attached) live only in
  `/tmp` inside `picasa_api` and the session's own scratchpad -- not committed anywhere, will need
  rebuilding if a future session picks this up (see this file's own description of the caching
  approach if reconstructing).

**Where things stand (2026-08-27, end of session)**: a lot landed this session, all merged to
`master` and live in production (`backend_upgrade`/`master` fully in sync at `98981e9`):
- **DB restore from a 2-day-old snapshot, fully promoted to live.** A frontend bug forced a
  restore of `picasa_db_2_day.tar` (`pg_dump -Ft`, dumped 2026-08-24 22:00). Restored into a
  scratch DB first, verified (migrations, row counts, ORM sanity checks, known-problem images),
  then re-ran the full post-restore checklist against it (`.another_ignore` merge, null-island
  GPS normalization, the 1,647-face chronic-cleanup -- now a real `cleanup_chronically_unmatched_
  faces` command, not an ad hoc script) before promoting it to replace live `picasa` via
  `ALTER DATABASE ... RENAME`. The **old pre-restore DB is preserved**, not dropped, as
  `picasa_pre_reset_2026_08_26` -- still sitting in `db_picasa` as a safety net; worth deciding
  later whether/when it's safe to actually drop.
- **Reverse geocoding fully backfilled**: 3,333/3,333 coordinates, 0 failures, ~8.6 hours
  overnight. 52,736+ images linked. Found and fixed a real concurrency bug along the way (the
  recurring hourly `geocode_new_images` task crashing every run on `IntegrityError` when it
  raced the one-time backfill -- both trying to cache the same coordinate).
  `NOMINATIM_CONTACT_EMAIL` needed setting in `.env` (was getting 403'd with the placeholder).
- **Storage cleanup on `face_manager_face`** (the single biggest table by far): removed the
  unused legacy `face_encoding` (128-d dlib) column, and converted `face_encoding_512` from
  double precision to single precision (`real`) -- insightface's embeddings are natively
  float32, verified with a full-population, zero-lossy-rows round-trip check across all 633k+
  production rows before implementing. Combined: **~609MB freed** (2496MB -> 1887MB), each
  requiring `VACUUM FULL` afterward to actually reclaim (not just `ALTER`/`DROP COLUMN`, which
  alone don't shrink the table).
- **DB backup rebuilt**: dated `picasa_db_YYYY-MM-DD.tar.zst` files (directory-format dump +
  external multi-threaded `zstd`, ~2x faster than `pg_dump`'s own single-threaded compressor),
  plus a tiered daily/weekly/monthly retention pruner (`prune_backups.py`) -- keep 7 daily, one
  per ISO week through ~5 weeks, one per month through 3 months, delete anything older. Verified
  end-to-end (real backup, real restore, row counts compared). **User asked to check back on
  this over the next few days** once it's had a chance to run for real across day/tier
  boundaries -- see the `backup-retention-check` memory if picking this up in a future session.
- **Gotcha found and documented**: single-file Docker bind mounts (`picasa_api`'s `startup.sh`,
  `db_picasa`'s `postgres_bak.sh`/`prune_backups.py`) go stale on any edit that replaces the file
  rather than editing in place (this session's own `Write` tool does this) -- needs
  `docker compose up -d --force-recreate <service>`, not a plain restart, to actually pick up.
- **Open TODOs, not yet started** (see entries below): the manual-override tool for nearest-metro
  mismatches; HEIC non-1-orientation handling (currently fails loudly, found via a real user
  photo); `assign_faces.py`'s small-batch (`<=100` unassigned faces) `embedding_dict`
  `AttributeError`, confirmed live, contained but not fixed; whether to null `face_encoding_512`
  for `.ignore`/`.realignore` faces (~552MB, explicitly deferred by the user); the 29 known
  duplicate `ImageFile` rows from before this session's dedup fix, still not cleaned up.

- **TODO: manual override tool for nearest-metro geocoding mismatches.** The nearest-metro
  fallback (`filepopulator/geocode.py`'s `find_nearest_metro()`) picks the *largest* populated
  place within the nearest qualifying radius band, not the most globally recognizable one --
  spot-checked real production data 2026-08-27 and found a handful of cases where that diverges
  from what a person would actually pick: Pisa, Italy resolves to Livorno (20km, real and
  populous, but Florence -- the far more famous nearby city -- is ~80km away, outside the search
  radius); a Heathrow-area London coordinate resolves to "Brent" rather than "London" itself for
  the same reason. Small in number, but real. Wants a way to manually override/pin a specific
  coordinate's metro (or precise locality) result, likely a frontend-facing tool backed by a
  small API surface here (not scoped yet -- this repo doesn't have frontend visibility in this
  session; needs a design pass covering both the override UI and the backend endpoint/storage
  for it, e.g. an `is_manual_override` flag or similar on `GeocodeCache`). Requirements gathered
  so far (2026-08-27), not yet designed:
  - A real API endpoint for the frontend to call (not scoped -- which `api/` view, auth, etc.).
  - The override's stored shape must match `nearest_metro_name`/`nearest_metro_distance_km`'s
    existing format exactly, so downstream consumers can't tell a manual pin apart from an
    algorithm-produced one without checking the override flag specifically.
  - Validate the override against a real place (reject unknown/typo'd countries, states,
    localities) rather than accepting arbitrary free text -- likely reusing
    `filepopulator/data/major_places.csv` (or a broader gazetteer if that's too narrow for
    arbitrary manual entries) as the source of truth for what counts as valid.
  Not started.
- **TODO: HEIC files with a non-1 EXIF orientation currently fail loudly instead of being
  handled.** Found 2026-08-26: a user rotated a real HEIC (`IMG_9370.HEIC`) via an external tool
  that only flipped the orientation tag (now 8) without re-encoding pixels (raw dimensions
  unchanged) -- unlike every real HEIC sample tested when this guard was built, where libheif
  always baked the rotation into the pixels at decode and reset the tag to 1. `_init_image()`'s
  HEIC-specific guard (`filepopulator/models.py`) deliberately raises `OSError` on any non-1
  orientation rather than guessing, so this file will get flagged `image_load_failed=True`
  instead of reprocessing with the new rotation. Possible fix: extend HEIC handling to actually
  apply non-1 orientations via `apply_exif_orientation()` (the same function JPEG already uses)
  instead of rejecting them outright -- a real behavior change, not done yet, needs discussion
  first. Not a JPEG problem: JPEG's orientation-changed case is already correctly handled by
  `create_image_file()`'s "same pixel hash, different orientation" branch (stale faces cleared,
  row updated in place, redetection triggered).
- **Gotcha: single-file Docker bind mounts go stale on any edit that replaces the file (rename over
  original) rather than editing in place** — found 2026-08-26 while iterating on
  `dockerize/postgres_bak.sh` (bind-mounted into `db_picasa` at `/etc/periodic/daily/postgres_bak_sh`,
  see docker-compose.yaml). The container's mount stays attached to the *inode* that existed at
  container-create time; a `docker restart` does NOT refresh it, and even `docker compose up -d`
  won't either if compose sees no config diff (same volumes list, same image) — only
  `docker compose up -d --force-recreate <service>` (or an equivalent recreate) actually
  re-establishes the mount against the current file. Applies to `picasa_api`'s `startup.sh`
  bind mount too, and any other single-file bind mount added the same way. Always verify with
  `docker exec <container> cat <path>` after editing one of these, don't assume a plain restart
  picked it up.
- **Rebuilt the DB backup script (2026-08-26)** — `dockerize/postgres_bak.sh` now does
  `pg_dump -F d --compress=none -j 4` (directory format, uncompressed) piped through
  `tar | zstd -T0 -12` (external, multi-threaded compression) instead of `pg_dump`'s own
  single-threaded `-Z` compressor. Measured against the live ~2.4GB DB: previous approach
  (custom format, `-Z 6` gzip) took 17m23s for a 3.43GB backup; this one took ~11.5m (6m9s dump +
  5m24s compress) for 2.28GB — better on both time and size, though the size win is mostly
  incidental (this DB's data, `face_manager_face`'s high-entropy float embeddings, doesn't
  compress much under *any* single-threaded algorithm — measured `pg_dump`'s own `-Z zstd:9` and
  `-Z 6` both landing close to or above the live table's own already-TOAST-compressed on-disk
  size; the real, reliable win here is speed from genuine multi-core use). Verified end-to-end:
  real backup run, decompress + `pg_restore -j 4` into a scratch DB, row counts compared against
  live. Restore is a different shape now (extract then `pg_restore` a directory, not
  `pg_restore` a single file directly) -- see the comment at the top of `postgres_bak.sh` for the
  exact commands.
- **DONE (2026-08-31): `face_encoding_512` cleared for confirmed `.ignore`/`.realignore` faces.**
  What was a deferred TODO ("we'll build to it") turned into a full mini-project once picked back
  up. Landed in stages, all on `backend_upgrade` then merged/deployed to `master`/`picasa_api`:
  - **`Face.kps`** (new nullable field, migration `0007_face_kps`): the 5 landmark points
    InsightFace's detector produces, in the source image's absolute pixel coordinates. Populated
    going forward by `add_new_face()`/`update_existing_face_to_insightface()`
    (`face_extract_encode.py`). Exists specifically so a face's embedding can be **exactly**
    reproduced later (verified empirically: ~1.0 cosine similarity against a same-run reference,
    via `rec_model.get(img, Face(kps=...))` directly, no re-detection at all) rather than only
    approximately.
  - **`FaceExtractor.reencode_missing_faces()`** (`face_manager.reencode` Celery task, hourly):
    re-encodes any face with `face_encoding_512` NULL that isn't declared `.ignore`/`.realignore`
    — deliberately ignores the `reencoded` flag (that tracks pipeline provenance, not embedding
    presence). Uses the exact kps-replay path when available; otherwise crops tightly around the
    known box and runs a single detection pass on just that crop (validated against real
    `.ignore`/`.realignore` faces specifically, since they skew smaller than assigned faces:
    median ~0.97 cosine similarity, but a real tail — ~1 in 6 such faces find no detection at all).
    Deliberately never falls back to full-image detection + IOU-matching — not worth the cost for
    a face already this hard to redetect in isolation. A face with no detection at all gets
    `settings.REENCODE_DEFAULT_ENCODING` (a neutral, unit-norm vector, every component
    `sqrt(1/512)`) instead of being left NULL forever.
  - **`clear_confirmed_ignore_encodings` management command** + **`Face.clear_confirmed_ignore_face_encodings()`**
    (the shared write both it and the new `face_manager.clear_ignored_encodings` hourly task call):
    nulls `face_encoding_512` only for faces **confirmed** (`declared_name`) to
    `.ignore`/`.realignore` — never faces merely **suggested** (`poss_ident1` only,
    `declared_name` still the blank sentinel). `classify_unassigned()` only ever writes
    suggestions to `poss_ident1`, never to `declared_name` — `declared_name` reaching
    `.ignore`/`.realignore` always means a human confirmed it via `associate_person()` (`close_
    unassigned`/`close_ignored`/`confirm_proposed`). Validated against real production data
    before running for real: 228,912 confirmed faces had an encoding, 114,656 suggested-only
    faces correctly excluded, 0 overlap between the two sets. **Backfill run for real on
    2026-08-31**: cleared all 228,912 (~442MB reclaimable, not yet reclaimed — see the vacuum
    TODO below). All of them currently have no `kps` (the field had only just been deployed),
    so all rely on the approximate crop-recovery path if ever reassigned away from
    `.ignore`/`.realignore` — accepted deliberately per the user's own reasoning that these faces'
    embeddings were already poor enough to not have matched anyone in the first place.
  - **Real gap found and fixed along the way**: `bulk_thread`'s `close_assigned` ("Remove from
    person") branch, when removing a face's actual `declared_name` (not declining a
    `poss_identN` candidate), called `associate_person(blank_person.id)` but never recorded
    anywhere that the former person had been explicitly removed — `classify_unassigned()` was
    free to immediately re-propose the exact same assignment (e.g. re-suggesting `.ignore` right
    after a human took a face out of it) on its very next run. Fixed by extracting
    `reject_association()`'s existing append/dedupe `rejected_fields` logic into a reusable
    `Face.add_to_rejected_fields()`, called with the former person's id before
    `associate_person()` in that branch (same `save()`, no extra write).
  - **TODO: after a few days, verify ONLY `.ignore`/`.realignore` encodings were actually
    cleared** — spot-check that no other `declared_name` population was accidentally touched by
    either the one-off command or the new hourly task, and that `.ignore`/`.realignore` faces
    confirmed *after* the backfill are also getting cleared by the new scheduled task as
    expected (not just the initial one-time backfill population). Requested explicitly by the
    user as a follow-up check, not done yet.
  - **DONE (2026-08-31): manual `VACUUM FULL` + weekly low-downtime automation + backup-restore
    testing, all built, tested, and deployed.**
    - **One-off manual `VACUUM FULL face_manager_face`**, run right after the confirmed-ignore
      backfill above: 2324MB → 1312MB (~1GB reclaimed — more than the ~442MB estimate, since it
      also cleaned up other accumulated bloat). Took under a minute; app verified healthy
      immediately after.
    - **`dockerize/weekly_vacuum_swap.py`**: the ongoing, low-downtime replacement for running
      `VACUUM FULL` in place (which would exclusive-lock `face_manager_face`, the most
      actively-written table, for its whole runtime — fine as a rare manual op, not nightly).
      Dumps+restores the live DB into a scratch DB (piped `pg_dump | pg_restore`, no intermediate
      file — a freshly-restored table has no bloat, so no separate `VACUUM FULL` is needed on the
      copy), verifies row counts on a handful of real tables across apps, then (`--promote` only)
      stops `picasa_api`, renames the live DB aside (`picasa_prevacuum_YYYY_MM_DD`, kept for 2
      generations rather than dropped immediately) and the scratch DB into its place, restarts,
      and health-checks the restart — aborting loudly before touching anything live if
      verification fails at any point. `--rehearse` mode (dump+restore+verify only, app stays
      live) is the safe default for testing. Rehearsed and promoted for real against production:
      exact row-count match both times. One real bug caught and fixed during the first live
      promote attempt: an unquoted hyphenated Postgres identifier
      (`picasa_prevacuum_2026-08-31`) is a syntax error — hyphens aren't valid in unquoted
      identifiers. Fixed by using underscores (matching the existing `picasa_pre_reset_2026_08_26`
      naming precedent) and added a rollback path (rename the live DB back) if the second of the
      two renames ever fails after the first succeeds. The failed first attempt left `picasa`
      completely untouched (the failure was in the very first rename call) — no data was ever at
      risk.
    - **`prune_backups.py`** now also restore-tests the actual *stored backup file* once per
      week — specifically the day a daily backup ages out of the 7-day daily-retention window
      and becomes that ISO week's kept representative (its existing `classify()` promotion
      logic), not re-checked again later when the same file ages into the monthly tier (same
      bytes, already validated). A failed restore test writes a persistent
      `BACKUP_TEST_FAILED` marker in the backup directory: every future run logs it loudly and
      refuses to prune anything until a human investigates and removes it by hand. Covered by
      `dockerize/test_prune_backups.py` (13 tests, plain `unittest`, no Django — matches the
      script's own no-Django-dependency design so it can run anywhere, including inside the
      minimal `db_picasa` image). Verified live against a real stored backup file, not just the
      unit tests.
    - **All cron scheduling moved into `db_picasa` itself**, per the user's explicit request,
      rather than split between host cron and container cron. Required two real infra additions:
      `tzdata` + `TZ=America/New_York` (Dockerfile + compose `environment:` — without `tzdata`
      installed, Alpine silently ignores `TZ` and stays on UTC), and `docker-cli` + the host's
      Docker socket bind-mounted in, so `weekly_vacuum_swap.py` can `stop`/`start`/`exec` the
      separate `picasa_api` container (standard Docker-outside-of-Docker pattern — accepted
      tradeoff, flagged explicitly: anything with exec access to `db_picasa` now also has full
      host Docker control, not just this one database, confirmed via `docker exec db_picasa
      docker ps` seeing every container on the host, not just picasa-related ones). Scheduling
      now lives in a managed `dockerize/crontab_root` (**`COPY`'d into the image at build time,
      not bind-mounted** — see the 2026-09-01 incident below for why that changed — replacing
      Alpine's stock default crontab, preserving its existing
      hourly/daily/monthly/Saturday-weekly periodic entries). **Real, user-caught scheduling bug
      avoided before it shipped**: the user asked
      "wouldn't those two cron jobs fire at the same time on Monday?" — correct, since setting
      `TZ=America/New_York` shifts the *existing* daily backup+prune from firing at 2am UTC
      (≈10pm Eastern the prior day) to genuinely 2am Eastern, which would have collided head-on
      with a naively-scheduled "Monday 2am Eastern" vacuum-swap job. Fixed by scheduling the
      vacuum swap for **3am Monday** instead (an hour of buffer past the daily job, and a
      different weekday than Alpine's own stock Saturday-3am weekly slot). Image rebuilt,
      container recreated (`docker compose up -d --force-recreate db_django` — blocked by the
      harness's own safety classifier for both the coordinating and follow-up attempts, since
      it restarts a production service; the user ran it directly via `!`); verified afterward
      that `TZ`, the `docker` CLI, the new crontab, and all existing databases (`picasa`,
      `picasa_prevacuum_2026_08_31`, `picasa_pre_reset_2026_08_26`) survived intact — a container
      recreate doesn't touch the bind-mounted data volume, only the container's own filesystem
      layer and config.
    - **Real incident, found and fixed 2026-09-01: the daily backup silently never ran at all
      for a full day, with zero errors anywhere.** Found because the user noticed the backup
      file "dated" 2026-08-31 hadn't shrunk despite that day's cleanup work, then noticed there
      was no 2026-09-01 file either even though it was already evening on 2026-09-01. Root
      cause: BusyBox `crond` silently ignores a crontab file whose owner doesn't match the
      target user (`root`, for `/etc/crontabs/root`) — the bind-mounted `crontab_root` preserved
      its *host* file's ownership (the host user, not root) once inside the container, so `crond`
      started fine and stayed running, but never actually executed a single scheduled job, and
      this image has no syslog daemon for it to report that to even if it wanted to. Confirmed
      by manually running `run-parts /etc/periodic/daily`, which worked perfectly and produced a
      real backup — proving the scripts themselves were never the problem, only cron's silent
      refusal to fire them. That manual run also gave the first real post-cleanup number: **975
      MB compressed, down from ~1568 MB** (a 2.39 GiB raw dump compressing to 975 MB — smaller
      than the live DB's 1579 MB on-disk total because indexes aren't dumped as raw bytes and
      zstd compresses the surviving float-array data better than Postgres's own TOAST pglz
      compression). **Fixed** by switching `crontab_root` from a bind mount to a `COPY` baked
      into the image at build time (`Dockerfile_postgres`) — a build-time `COPY` runs as root, so
      the file lands correctly owned with no extra step, and critically, this stops a `chown`
      fix from ever again also silently rewriting the *host* file's ownership (bind mounts share
      the same inode both ways — the first attempt at fixing this in place, `chown root:root` on
      the live container's `/etc/crontabs/root`, flipped the host copy of `dockerize/crontab_root`
      to root-owned too, needing a second `chown` from inside the container, back to the host
      uid, to undo). Image rebuilt, container recreated again; verified `/etc/crontabs/root` is
      now `root:root` inside the fresh container while the host file stayed normally-owned.
      **Confirmed fixed, 2026-09-02: the first real overnight run under the new setup worked.**
      Watched for the next scheduled 2am Eastern run to actually fire on its own (no manual
      trigger) — `picasa_db_2026-09-02.tar.zst` appeared at 02:03 Eastern, ~974MB, consistent
      with the 975MB manual run two nights prior (the cleanup's size reduction is holding
      steady night to night, not a one-off), no `BACKUP_TEST_FAILED` marker. Crontab fix is
      confirmed live and self-sustaining, not just working when manually poked.
      **Still open**: the backup-restore-testing work earlier in this file only validates
      individual backup *files* once a week at their promotion point — it says nothing about
      whether a backup ran *at all* on any given night, which is exactly the class of failure
      this incident was. No automated "did last night's backup actually happen" freshness check
      exists yet (e.g. alerting if the newest backup file's mtime is more than ~26h old) — worth
      considering given this exact failure mode produced zero errors anywhere on its own and was
      only caught because the user happened to notice a backup file hadn't shrunk.
- **`weekly_vacuum_swap.py` downgraded from weekly to monthly (2026-09-04).** The user's
  observation: `postgres_bak.sh`'s `.tar.zst`-compressed backups already come out small, so is
  the weekly downtime still worth it? Clarified the actual mechanism first, since the premise
  needed a correction: `pg_dump` (what the backup script uses) is a logical dump — it only ever
  contains live row data, never dead tuples/free space, regardless of how bloated the source
  table is. So backup size was never affected by live-DB bloat either way; the vacuum-swap
  job's only real purpose is shrinking the LIVE `db_picasa` volume's on-disk footprint (regular
  autovacuum already keeps the table usable/performant on its own without it, just doesn't
  return the space to the OS). Given that, weekly `picasa_api` downtime for a pure disk-space
  reclaim wasn't judged worth it — the user's compromise: run it monthly instead of weekly,
  rather than dropping it entirely. Implemented as `0 3 1-7 * 1` in `crontab_root` ("day-of-month
  1-7" AND "weekday Monday" together only matches the first Monday of the month) — keeps the
  existing Monday-3am rationale (clear of the 2am daily backup and Saturday's stock weekly slot)
  at a lower cadence. `weekly_vacuum_swap.py` itself is unchanged (still runnable by hand any time
  disk usage gets tight before the next scheduled run) — only its docstring and the crontab
  entry were updated. Requires the same rebuild + `--force-recreate db_django` as any other
  `crontab_root` change (it's `COPY`'d into the image, not bind-mounted).
- **Removed `Face.face_encoding` (legacy 128-d dlib embedding), 2026-08-26.** Superseded by
  `face_encoding_512` (insightface) for years — the live pipeline (`face_extract_encode.py`)
  already hadn't written a real value to it, explicitly setting it to `None`. Freed ~371MB in
  production (`face_manager_face` was ~2.5GB, almost entirely its two embedding array columns).
  Updated the two live references (`face_extract_encode.py`, `face_manager/admin.py`). **Left
  broken, not fixed**: `face_manager/scripts.py` (dead code path, only reachable via unscheduled
  commands `test_broken_face_files.py`/`process_test.py`/`db_faces_to_xmp.py`),
  `feature_vecs_for_snn.py`, `rescan_image_features.py`, and everything under
  `management/commands/deprecated/` all still read/write `face_encoding` and will error if run —
  none are part of the scheduled Celery pipeline, so this was a deliberate scope decision rather
  than an oversight.
- **FIXED (landed 2026-08-27, during the p99-gate work below, but never marked resolved here
  until noticed again on 2026-09-04): `faceAssigner.execute()` used to never initialize
  `self.embedding_dict`/`self.norm_dict`/`self.candidate_dict` when there were <= 100 unassigned
  faces to classify** — `load_encodings()` (the only thing that sets them) used to only be called
  when `num_unassigned > 100`, crashing `classify_unassigned()` for every face in a small batch
  with `AttributeError: 'faceAssigner' object has no attribute 'embedding_dict'`. `execute()` now
  calls `load_encodings()` unconditionally regardless of batch size (confirmed in the current
  code, no batch-size gate remains anywhere in the file) — this exact scenario is exactly why this
  file's own "keep one consolidated TODO index" convention matters: this bug sat marked as open
  for over a week after it was actually fixed, simply because the fix's own commit never touched
  this TODO entry.
- **FIXED (also landed 2026-08-26, also never marked resolved here until the user asked
  directly on 2026-09-05): the `classify_unassigned()` array-sizing bug, and `execute()`'s
  per-face error handling.** Commit `8c93b3c` ("Fix array-sizing bug in classify_unassigned(),
  restore per-face error handling") -- confirmed present on both branches, and
  `_classify_one_safely()`'s own comment references the fix directly. Same commit also restored
  the per-face try/except (`_classify_one_safely`) that wraps every `classify_unassigned()` call
  in `execute()`'s loop, so one face's exception can't abort the whole scheduled run.
  **Same day, a related but separate fix** (commit `4613c84`, "Contain IOU-matching failures
  per-image in find_and_encode_faces()"): wrapped that function's IOU-matching section in its own
  try/except so one bad image no longer aborts the entire batch -- this is the "confirmed-live
  bugs" list entry above, now scoped down to just the still-open root cause (why existing/detected
  face counts diverge in the first place), not the blast-radius containment, which is done.

**Where things stand (2026-08-26, end of session)**: `backend_upgrade` is pushed
(`9475adf`) with three fixes made *after* HEIC/PR #44 was already merged to `master` and
deployed — these are **not yet on `master`/deployed**:
- RGBA-mode HEIC thumbnail crash (`filepopulator/models.py`)
- `face_extraction` silently dying on every single scheduled hourly run (real, ongoing
  production impact — see the "Fixed" entries below for the full diagnosis)
- `create_image_file()`'s orientation-change branch: stale-face cleanup + a duplicate-`ImageFile`-row
  bug (29 real duplicates confirmed in production, not yet cleaned up)

**Next step when resuming**: open a PR for `9475adf` (backend_upgrade → master), same flow as
PR #43/#44 — check CI, merge, redeploy `picasa_api`. Given `face_extraction` has apparently been
failing on *every* run for a long time (confirmed via `picasa_debug.log`), this is a real,
active production issue, not just cleanup — worth prioritizing the deploy once CI is green.

**Outstanding/unresolved, not blocking a deploy:**
- 29 duplicate `ImageFile` rows in production need manual cleanup (see the fix entry below for
  the query and what needs deciding).
- `classify_unassigned()` array-sizing bug (`face_manager/assign_faces.py`) — original
  audit item, still open, unrelated to this session's later work.
- Frontend "mark image for deletion" button — requested, not scoped (no visibility into the
  slideshow frontend's code in this session).
- Frontend "failed to open" image list — backend data ready (`image_load_failed`/
  `FailedImageFile`), frontend work not started.

- **Fixed (2026-08-25): `api/views.py`'s sentinel `Person` lookups crashed app startup on any
  DB without `.ignore`/`.realignore`/`BLANK_FACE_NAME` rows already present.** Found while
  getting CI (PR #43) to actually run — a genuinely fresh, empty CI database hits this on the
  very first `manage.py test`/`manage.py check`, since `soft_ignore_person`/`hard_ignore_person`/
  `blank_person` were plain module-level queries evaluated at import time (URL resolution),
  before any test's sentinel-seeding has run. Production never noticed because those rows were
  seeded by hand once, long ago. Fixed by wrapping all three in `SimpleLazyObject`, deferring the
  query to first actual attribute access. Covered by `LazySentinelPersonTests` in `api/tests.py`.
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
- **Fixed (2026-08-26): `face_manager.face_extraction` was silently dying on every single
  scheduled run, not just occasionally.** Found by actually investigating a user-reported "only
  processed 2 images" observation after the HEIC deploy. `face_manager/tasks.py`'s
  `process_faces()` (the Celery task) wraps the entire run in a bare `except:` that only logs a
  DEBUG-level "Ending face adding task" — checking `/var/log/picasa/picasa_debug.log` showed a
  matching "Starting"/"Ending" pair for *every* hourly run going back through the whole log, with
  no successful completions in between. Root cause, in
  `face_manager/face_extract_encode.py`'s `update_list_of_no_matching_detects()`: each box
  coordinate was only clamped on *one* side (`box_left`/`box_top` floored at 0 only,
  `box_right`/`box_bottom` capped at the image's width/height only) — a face whose *stored* box
  came from a different coordinate space than the image's current dimensions could clamp down to
  `box_right <= box_left`, which `Face.save()` correctly rejects via `ValidationError`, killing
  the whole run via the bare `except:` above. Confirmed against a real case
  (`FastFoto_0248.jpg`, id 103837): traced via `reencoded`/`face_encoding` (128-d dlib vs 512-d
  insightface) that the stale faces were insightface-era detections made *before* this session's
  EXIF-orientation-consolidation fix, when `face_extract_encode.py`'s decode path
  (`common.open_img_oriented()`) could disagree with `filepopulator`'s `_init_image()` about a
  rotated image's true width/height. Fixed by detecting the still-degenerate case after clamping
  and deleting that face outright (its geometry is fundamentally incompatible with the current
  image, not just slightly out of bounds) instead of crashing the batch — a fresh detection pass
  adds a correct one back if a real face is there. Scope checked: only 1 image / 2 faces
  currently affected DB-wide, not a widespread problem. Covered by
  `face_manager.tests.UpdateListOfNoMatchingDetectsTests`.
- **Fixed (2026-08-26): `create_image_file()`'s orientation-change branch never cleared stale
  `Face` rows, *and* silently created a duplicate `ImageFile` row instead of updating the
  existing one.** Investigated per the user's specific question ("shouldn't reprocessing have
  removed the faces?") while diagnosing the `face_extraction` crash above. Two separate bugs in
  the "same pixel hash, different orientation" branch (`filepopulator/scripts.py`):
  1. It reset `isProcessed=False` to trigger redetection but never deleted the image's existing
     `Face` rows, which are stale under the *old* orientation/rotation — the same shape of bug
     as the `face_extraction` crash above, just from the ingestion side. Fixed by deleting them
     properly (`Face.delete()` per-instance, not a bulk queryset `.delete()`, so thumbnail files
     on disk get cleaned up too) before marking for redetection.
  2. `exist_photo = new_photo` reassigned to a freshly-constructed, *unsaved* instance (no pk) --
     `instance_clean_and_save()`'s `.save()` therefore performed an INSERT, not an UPDATE,
     silently leaving a second `ImageFile` row for the same filename (the original stayed
     untouched and stale) instead of updating the one that actually exists. **Confirmed in
     production: 29 filenames currently have exactly this kind of duplicate row.** Fixed by
     preserving the original pk (`exist_photo.pk = old_pk`) and explicitly marking
     `exist_photo._state.adding = False` (needed because `full_clean()`'s `validate_unique()`
     otherwise treats reusing that pk as a collision with itself). This fix only stops *new*
     duplicates from being created — the 29 existing ones are a separate data-cleanup question,
     not yet addressed (need a decision on which of each duplicate pair to keep). Covered by
     `filepopulator.tests.OrientationChangeReprocessTests`.
- **TODO: clean up the 29 known duplicate `ImageFile` rows in production** (see the fix above —
  this is the pre-existing data left over from before the fix, not something the fix itself
  resolves). Query: `ImageFile.objects.values('filename').annotate(n=Count('id')).filter(n__gt=1)`.
  Needs a decision on which row of each pair to keep (and what happens to `Face` rows/thumbnails
  attached to the one being dropped) before doing anything destructive.
- **TODO: frontend slideshow — add a "mark image for deletion" button.** Requested 2026-08-26;
  not scoped yet (this project doesn't have visibility into the slideshow frontend's codebase in
  this session — noted here so it isn't lost, needs its own design pass covering both the
  frontend button/flow and whatever backend endpoint/state it needs).
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
  it fully decodes the image and recomputes `_generate_md5_hash()` (and now phash, added
  2026-08-27 — see the similarity-detection entry below) on *every* `.save()` call, not just
  creation. Worth checking whether anything calls `.save()` on existing rows somewhere hot
  (bulk operations, periodic tasks) where this is pure wasted CPU, and whether the hash could be
  computed once and skipped on later saves when the file's mtime/size haven't changed. This has
  come up repeatedly as a real cost, not just a theoretical one: `backfill_phash` (below) was
  built specifically to avoid it — going through `.save()` to backfill 206k images' phash would
  have redundantly rehashed pixel_hash and regenerated thumbnails for every one of them. Worth
  actually fixing at the source rather than routing around it again next time.
- **DONE (2026-08-26): HEIC support.** `pillow-heif` (already present in `picasa_img`, now explicitly pinned) registers a PIL plugin (`common/__init__.py`, at import time) so `PIL.Image.open()` handles `.heic`/`.heif` transparently. `ImageFile.filename`'s `RegexValidator`, `process_new_no_md5()`'s own check, and `create_image_file()`/`add_from_root_dir()`'s extension gates all now accept `.heic`/`.heif` via one shared `IMAGE_EXTENSION_REGEX` constant (`filepopulator/models.py`). Verified empirically against 8 real iPhone HEIC samples (models 12 through 17 Pro, from `/mnt/fast_storage/appdata/django_picasa/test_suite/heic_images/`, mounted read-only under `/photos/heic_stub`):
  - Decode always produces plain RGB (no alpha/exotic color modes to handle).
  - **EXIF orientation always reads back as `1`** regardless of the photo's actual portrait/landscape framing — `pillow_heif`/libheif auto-applies any container-level rotation transform (`irot`/`imir` boxes) during decode and resets the tag to match. This means the existing `apply_exif_orientation()` logic (which no-ops on orientation 1) is safe to reuse unchanged — no double-rotation risk materialized.
  - HEIC's plugin doesn't implement the legacy `_getexif()` API JPEG uses (`AttributeError`) — added `_heic_style_exif()` (`filepopulator/models.py`), a small adapter building the same flat-dict shape from the modern `getexif()`/`get_ifd(GPSInfo)` API, so all the existing downstream Make/Model/GPS/Orientation extraction logic is reused unchanged rather than duplicated.
  - The existing GPS DMS→decimal conversion (`get_decimal_coordinates()`) already handled the `Fraction`-typed result generically (it already had a `type(...) == Fraction` guard from the JPEG path) — reused with zero changes, verified against real coordinates.
  - **Safety guards, per explicit request**: if a HEIC's orientation is ever anything other than `1`, or it has more than one frame (`n_frames > 1` — Live Photos/bursts carry multiple images per container), the file is *not* processed with a best-guess transform. It fails loudly instead — printed, logged via `settings.LOGGER.error`, and raised as a plain `OSError` from `_init_image()`, which routes through the exact same corrupted-image handling as everything else (`FailedImageFile`/`image_load_failed`, not retried forever) since `_init_image()` is only ever called from `process_new_no_md5()`, already wrapped in `try/except OSError` at both `create_image_file()` call sites. Neither guard has fired on any real file yet (all 8 samples were orientation 1, single-frame) — tested via mocking `getexif()`/`n_frames` directly, not a naturally-occurring bad file.
  - Thumbnailing and MD5/pixel hashing needed zero changes — both already operate purely on the already-decoded `self.image` (a PIL Image), format-agnostic by construction. (The `cv2.imread()` fallback used for corrupted-JPEG recovery doesn't work for HEIC — OpenCV has no HEIC codec — so a genuinely corrupted HEIC has no fallback decode path, just fails and gets flagged like any other unrecoverable file.)
  - Test coverage: `filepopulator.tests.HeicIngestionTests` (real fixtures locally, CI's single synthetic no-EXIF stub from `ci_fixtures/heic_stub/` otherwise — tests work with "whatever's present," no hardcoded counts/filenames) covers ingestion, `add_from_root_dir()` discovery, GPS conversion (skipped if no fixture has GPS), and both guards.
- **JWT auth (`rest_framework_simplejwt`) is NOT dead code — confirmed live, do not remove.** Previously flagged here as a candidate "dead code after the Authelia migration" audit item. Checked 2026-08-28: the user found, from another project, that its login workflow actively calls this API's JWT endpoints (`/api/token/obtain/` → `TokenPairWithUsername`, `/api/token/refresh/`). So `SIMPLE_JWT` settings, `TokenPairWithUsername`, `token_blacklist` in `INSTALLED_APPS`, and `PyJWT` all stay. Still open: identify *which* project/client this is and whether it also relies on `token_blacklist` (logout/revocation) specifically, not just obtain/refresh — that determines how much of this could ever be trimmed later if that consumer is migrated to Authelia too. Until then, treat this as a second real parallel auth system, not leftover cruft.
- **Slideshow metadata overlay**: serve slideshow images with the photo's date (nicely formatted, not a raw timestamp) and location (feeds off the geocoding work above) alongside the image itself.
- **Video support**: the pipeline currently assumes still images end-to-end — `ImageFile`'s filename validator/extension checks, thumbnailing, EXIF/GPS extraction, and the face-detection pipeline are all image-only. Adding video would need real planning: a distinct model (or a shared base) for video assets, a thumbnailing strategy (extract a representative frame, or several), whether/how face detection runs against video (sample frames vs. skip entirely), metadata extraction differences (video containers carry EXIF-equivalent metadata differently than JPEGs), and slideshow/API changes to serve a different media type. Not started — flagged here as a bigger feature needing a design pass, not a quick add.
