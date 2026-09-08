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

**Open questions / follow-ups:**
- No automated "did last night's backup actually run" freshness check exists — the current
  restore-testing only validates a backup file once it's promoted into weekly retention, which
  says nothing about a night the backup silently never ran at all (this has already happened
  once, caught only because the user happened to notice a file hadn't shrunk).
- A manual override tool for nearest-metro geocoding mismatches is wanted but not scoped (needs
  both a frontend UI and a backend endpoint/storage design).
- A looser classification threshold specifically for faces with `.ignore` already in their reject
  list was brainstormed but never validated or built.

**Bigger, deliberately unscoped features:**
- Slideshow metadata overlay (photo date + location shown alongside the image).
- Video support — design pass, Phase 0 (infra) and Phase 1 (`VideoFile` model + ingestion) landed
  2026-09-07. Phases 2-4 (thumbnailing, face detection, API) not started. See the dated write-up
  below for the full design and phased build plan.

**Frontend (out of scope for this repo — no visibility into that codebase from here):**
- "Mark image for deletion" button for the slideshow.
- "Failed to open" image list surface (backend data — `image_load_failed`/`FailedImageFile` — is
  ready; frontend work never started).
- A way to upload new files from the frontend, including zip archives (would need server-side
  unpacking before the normal ingestion path could pick them up) — requested 2026-09-07, not
  scoped (no existing upload endpoint on this backend at all yet — would need a new one designed
  alongside the frontend work).
- A status page — requested 2026-09-07, not scoped. `/api/server_stats/` (`StatsViewSet` /
  `ServerStatsSerializer`) already exists and surfaces some of this (image/face counts, percent
  processed, estimated time remaining) but was built for a different purpose; worth checking
  whether it's sufficient as-is or needs new backend fields once the frontend design is known.

(Resolved items -- fixed tests, the Django 6.1 upgrade, pruned-stale brainstormed ideas, etc. --
have been cleared from this index once actually done; their full write-ups remain in the dated
narrative below, findable by searching a distinctive word.)

**Video support — design pass + Phase 0 (2026-09-07).** Full design discussed with the user before
any code: a separate `VideoFile` model (not a shared base with `ImageFile` -- too many
photo-specific EXIF fields to drag along, and it'd force a risky refactor of every existing
`ImageFile` call site) mirroring `ImageFile`'s field *names* deliberately (`thumbnail_big`,
`filename`, `dateTaken`, etc.) so code can duck-type across both via a `Face.source_media` property
later. `Face` gets a second nullable FK (`source_video_file`, alongside the existing
`source_image_file`), not a `GenericForeignKey` -- two plain FKs keep `select_related`, the
covering indexes added earlier this session, and referential integrity intact; a `CheckConstraint`
(`Q(source_image_file__isnull=False) ^ Q(source_video_file__isnull=False)`) enforces exactly one
source per face. Crucially, **no primary-key harmonization is actually needed** --
`ImageFile.pk`/`VideoFile.pk` are separate sequences in separate tables, so there's no collision to
resolve, and everything identity-related (`Person`, `declared_name`, `poss_identN`, the whole
`assign_faces.py` classification pipeline, verification clustering) is already orthogonal to
"photo vs. video frame" and needs zero changes -- it just counts `Face` rows regardless of source.

**Metadata reality check, done before committing to any tooling**: exposure/aperture/ISO/focal-
length/flash/light-source have no standard video equivalent at all (a per-exposure photo concept;
video is a continuous stream, consumer devices essentially never embed per-clip camera settings) --
a real, structural loss, not a tooling gap. Orientation is recoverable but via a rotation matrix in
the container's track header, not an EXIF enum -- needs its own parsing, can't reuse
`apply_exif_orientation()`. GPS and creation-date are generally recoverable. Tooling split
deliberately two ways rather than one: `ffmpeg`/`ffprobe` for frame extraction and stream info (the
right tool for that), `exiftool` for the metadata specifically (GPS and camera-make/model in
particular are meaningfully more reliable out of exiftool's tag database than ffprobe's, across the
range of real camera/phone manufacturers) -- same shape as this project already uses PIL for
image decode and a separate GPS-conversion helper for EXIF GPS math. Real per-file verification via
`ffprobe`/`exiftool` against actual library samples is planned for Phase 1, before any parsing logic
is written (same discipline HEIC support used -- 8 real samples before committing to an assumption
-- skipping it is exactly how the original IOU-matching bug happened).

**Face detection phasing, deliberately NOT started yet (holding at the user's explicit request)**:
Phase 3 in the plan below is sparse frame-sampling through the *existing*, unchanged
`PyramidalDetector`, accepting near-duplicate faces per person per clip as a known MVP limitation
(the existing `verification_cluster_group` review tooling already helps here) rather than building
new cross-frame tracking machinery up front. Real tracking (collapsing same-face-across-frames
before it reaches classification, via the same `linear_sum_assignment` primitive this session's
IOU-matching rewrite already uses) is an explicit Phase-3.5, not Phase 1.

**Phased build plan**:
- **Phase 0 (infra, no app code) -- DONE, landed this session.** Real host directories confirmed to
  exist and be readable (`/mnt/data/samba_share/Video/{Our_Home_Videos,Lewis_family_videos}`) --
  the shared parent also has unrelated sibling folders (`Marco Polos`, `Movies`, `TV`, a loose
  `t.mp4`) the user does NOT want scanned, and doesn't want to reshape the directory layout right
  now to avoid them. Resolved as: **one bind mount of the shared parent**
  (`VIDEO_ROOT=/mnt/data/samba_share/Video` in `.env`, `${VIDEO_ROOT}:/videos:ro` in
  `docker-compose.yaml`) plus an explicit **`VIDEO_ROOTS` allowlist in settings**
  (`picasa/settings.py`) naming only the two wanted subfolders -- simpler than two separate mounts,
  and the unwanted siblings are just never walked since nothing outside the list is ever
  `os.walk()`'d. `FILEPOPULATOR_SERVER_VIDEO_DIRS = VIDEO_ROOTS + [PHOTO_ROOT]` also covers videos
  interspersed in the existing photo tree (confirmed some are). `ffmpeg` and `exiftool` (package
  `libimage-exiftool-perl`) added to `Dockerfile_picasa`, built and verified in a throwaway image
  (`ffmpeg`/`ffprobe`/`exiftool` all run correctly) before touching the real image. Real `.env` on
  the host updated with the new `VIDEO_ROOT` line (additive, harmless until the container is
  actually recreated with the new compose file -- not yet done, see below). Full fast suite:
  320/320 passing.
  - **Small unrelated bug fixed along the way**: `filepopulator/tasks.py`'s `load_images_into_db()`
    had its own `celery_app.control.inspect().active()` guard checking for a task named
    `'face_manager.populate_files_from_root'` -- a name that never existed (the task is actually
    registered as `'filepopulator.populate_files_from_root'`), so the check could never trigger.
    Removed entirely rather than just fixing the string, since it's also redundant:
    `add_from_root_dir()` already holds its own real Postgres advisory lock covering every entry
    point, the same reasoning that already removed an equivalent racy check from
    `face_manager.tasks.process_faces()`.
  - **DONE (2026-09-07): rebuilt `picasa_img` and recreated `picasa_api`** with the new mount and
    binaries. `docker compose build picasa` then `docker compose up -d --force-recreate picasa`
    (`db_picasa` was also recreated as part of the same compose operation -- verified data fully
    intact afterward, row counts matched). Verified live: `/videos` mount present and correctly
    shows the parent's full contents (including the unwanted siblings -- `VIDEO_ROOTS` is what
    actually scopes ingestion, not the mount), `ffmpeg`/`ffprobe`/`exiftool` all run correctly,
    `manage.py check` clean, all expected Celery tasks still registered, a `person_list` smoke
    test still returns 200.
- **DONE (2026-09-07): Phase 1 -- `VideoFile` model, ingestion, and scheduled task.**
  `FailedVideoFile` (mirrors `FailedImageFile` exactly) + `VideoFile`
  (`filepopulator/migrations/0007_failedvideofile_videofile.py`) -- thumbnail fields included now
  even though Phase 2 populates them, to avoid a second migration later.
  `VIDEO_EXTENSION_REGEX` sized against a **real extension survey** of the actual mounted
  directories (`find ... | sed 's/.*\.//' | sort | uniq -c`), not guessed: real video extensions
  turned out far broader than assumed (`mp4`, `mov`, `mpg`, `avi`, `m2ts`, `mts`, `wmv`, `3gp`/
  `3gpp`, `m4v`, `mkv`), and the same directories are full of non-video clutter that needed
  explicit exclusion -- `.bif` (2334 files, Roku trick-play thumbnails), `.modd`/`.moff` (JVC
  camcorder sidecar metadata), `.thm` (thumbnail sidecars).

  **Real per-file metadata verification, done before writing any parsing logic** (via `ffprobe`/
  `exiftool` against one real sample per extension) -- confirmed the design's assumptions and
  surfaced two things that changed the actual implementation:
  - `creation_time` is present via `ffprobe` for phone-shot formats (mp4/mov/3gp) but **absent**
    for camcorder/older formats (mpg/avi/wmv/m2ts/mts/m4v) -- a real, structural gap, not a
    tooling failure. `exiftool` recovered `DateTimeOriginal` **and** `Make`/`Model` for a real Sony
    `.MTS` file where `ffprobe` found nothing at all -- confirms the two-tool split was the right
    call. One real `.mpg` file had **no recoverable metadata whatsoever** (confirmed via both
    tools) -- a genuine permanent gap for some old digitized home movies, handled by falling back
    to `guess_date_from_filename()` (reused directly from the image pipeline -- many of these
    video filenames embed a capture timestamp the exact same way phone photo filenames do) and
    finally `timezone.now()` with `dateTakenValid=False`, the **same fallback order
    `ImageFile._get_date_taken()` already uses** (initially assumed a file-mtime fallback instead;
    checked the actual code and corrected course before implementing).
  - **A real, in-progress bug caught by testing against real files, not synthetic ones**: a global
    `exiftool -n` flag (chosen to force clean decimal GPS output) also disabled print-conversion
    for `Make`, and the same real Sony file has an undecoded numeric `Make` tag that came out as
    the meaningless raw code `264` under `-n` instead of the correct `"Sony"` string `exiftool`
    gives without it. Fixed with exiftool's per-tag `#` suffix (`-GPSLatitude#`/`-GPSLongitude#`)
    to force numeric output for *only* those two tags, leaving `Make`/`Model`/dates on their
    normal human-readable decoding.
  - **A second real bug, same discovery pass**: `dateutil.parser.parse()` was tried as the
    date-parsing fallback (for exiftool's `'2018:12:14 20:15:14-04:00 DST'` shape) and silently
    defaulted the date portion to **today()** while still parsing the time correctly -- no
    exception raised, so it wasn't merely untested, it was actively wrong and undetected until
    manually inspecting the output. Fixed by using explicit `strptime` formats (including a
    `%z`-aware one for the numeric-offset case) instead of dateutil, after stripping a trailing
    zone-name word (`"DST"`) neither format expects.
  - `Rotation` needed no override -- already a clean signed-degree integer from exiftool without
    `-n`, confirmed against the same real iPhone file that showed `ffprobe`'s own side-data
    rotation disagreeing in sign (90 vs. -90) -- exiftool is used as the sole source for rotation,
    never mixed with ffprobe's.

  **`filepopulator/video_scripts.py`** (`create_video_file()`, `add_videos_from_root_dir
  (root_dirs: list)` -- own advisory lock name `filepopulator.add_videos_from_root_dir`, loops
  `FILEPOPULATOR_SERVER_VIDEO_DIRS` under one lock acquisition rather than one call per root) +
  new `filepopulator.populate_videos_from_root` Celery task (own schedule, 30 minutes past every
  hour, offset from the photo scan; matches the existing one-task-per-concern pattern rather than
  folding into `load_images_into_db()`). **Deliberate scope cut, not an oversight**: no
  `delete_removed_videos()` equivalent yet -- a `VideoFile` row for a file that's since vanished
  from disk isn't cleaned up automatically.

  **Tested against all 8 real sample files** (one per real extension, staged at
  `/mnt/fast_storage/appdata/django_picasa/test_suite/sample_photos/video_samples/` -- the
  established real-fixture convention, reachable at `/photos/video_samples` the same way
  `heic_stub`/`corrupted` already are) with **zero failures** after the two bugs above were fixed,
  correct dates/GPS/camera-info/rotation recovered exactly where expected and absent exactly where
  the source genuinely has nothing embedded. `VideoIngestionTests` (11 new tests, `filepopulator/
  tests.py`) covers ingestion, root-discovery, re-run idempotency, multi-root walking, a
  nonexistent-file failure case, and GPS/camera-info recovery when present; a synthetic
  `ci_fixtures/video_stub/synthetic.mp4` (one-second solid-color pattern, no embedded metadata,
  generated via `ci_fixtures/generate_fixtures.py`'s new `build_video()`) backs the same tests in
  CI, which also gained `ffmpeg`/`libimage-exiftool-perl` in its system-library install step.
  `ParseExifVideoDateTests` (5 new tests) unit-tests the date-parsing fix directly, including the
  exact real-world string that broke `dateutil`. Full fast suite: 331/331 passing (320 baseline +
  11 new).
  - **DONE, same day: deployed to production and validated against real data end-to-end.**
    Sequenced migrate-then-restart (safe here since it's purely additive -- new tables never break
    currently-running old code, unlike the earlier column-drop case that needed the opposite
    order). Verified live: `manage.py check`/`migrate --check` clean, new task registered
    (`celery inspect registered`), and a real, non-synthetic smoke test -- ran
    `add_videos_from_root_dir()` against one real folder
    (`/videos/Our_Home_Videos/Phone/2025/thanksgiving`) rather than just the 8-sample set: **42
    real videos ingested with zero failures**, all `Apple iPhone 12 mini`, all correct dates
    (Nov 26-29 2025, ~19.7 minutes total), 41 of 42 with GPS. Left these as real, permanent data
    (not test artifacts to clean up) -- this is genuine partial backfill, not a throwaway check.
  - **DONE, same day: `MIN_VIDEO_DURATION_SECONDS` (default 3s) excludes short clips** (Live
    Photos/motion clips, accidental taps), per the user's request, added *before* running the full
    library backfill so short junk wouldn't need cleanup afterward. Checked via `ffprobe`'s
    duration before the more expensive `exiftool` call (cheaper to rule out a short file than a
    normal one), recorded via `FailedVideoFile` with an "Excluded: duration Xs is below the Ys
    minimum" message rather than silently skipped -- reuses the same mtime-check mechanism that
    already stops a real failure from being retried every scan, and keeps the exclusion visible/
    trackable rather than silently invisible. Not a hypothetical: one real fixture
    (`PICT0335.AVI`, an old digitized home-movie snippet, genuinely 2.1s) and the CI synthetic
    stub (1s) both already exercise this path, so `VideoIngestionTests` was rewritten to partition
    fixtures by actual measured duration rather than assume every fixture clears the bar -- real
    regression coverage, not fixtures chosen to avoid the new behavior. Full fast suite: 332/332
    (331 + 1 new exclusion-specific test).
- **IN PROGRESS right now (started 2026-09-07, ~15:04 America/New_York): full production video
  backfill running.** Kicked off manually (not waiting for the hourly schedule) via
  `docker exec picasa_api python manage.py shell -c "from filepopulator.video_scripts import
  add_videos_from_root_dir; from django.conf import settings;
  add_videos_from_root_dir(settings.FILEPOPULATOR_SERVER_VIDEO_DIRS)"`, running detached
  (`docker exec -d`, so no captured stdout log -- monitor via `VideoFile`/`FailedVideoFile` row
  counts and `docker exec picasa_api ps aux | grep manage.py` instead). Covers all three
  `FILEPOPULATOR_SERVER_VIDEO_DIRS` roots (`Our_Home_Videos`, `Lewis_family_videos`, and
  `PHOTO_ROOT` for interspersed videos) against the real library (~6,300+ real video files by
  extension count, thousands more `.bif`/`.modd`/`.moff`/`.thm` correctly excluded by extension).
  Progress snapshot at last check: 769 `VideoFile` + 168 `FailedVideoFile` (most likely the new
  duration-exclusion, not real failures -- not yet broken down). **When this finishes**: report
  final counts/stats to the user (duration exclusions vs. real failures, date range, camera
  makes/models seen, GPS coverage, total footage) -- this was explicitly why the user wanted the
  backfill run now instead of waiting on the schedule ("so we can get more info on the library").
  Nothing else is blocked on this finishing -- Phase 2 (thumbnailing) can start independently
  whenever revisited.
- **Phase 2 (not started)**: thumbnailing via one extracted `ffmpeg` frame, reusing the existing
  thumbnail-generation code unchanged once a PIL Image exists; handle the rotation matrix here.
- **Phase 3 (design/investigation started 2026-09-07, no code yet, holding on implementation at
  the user's request): sparse-sampled video face detection.** Original plan (superseded) was to
  reuse the existing `PyramidalDetector` as-is; the user instead wants a real single-pass
  detector for video (no multi-scale pyramid -- video's own frame-to-frame redundancy already
  covers what the pyramid buys for stills) combined with clustering (same complete-linkage
  approach as `verification_clustering.py`) plus IOU-based track-stitching (the same
  `linear_sum_assignment` primitive the image reconciliation rewrite already uses) to merge
  broken same-person tracks across a clip, then drop tiny/transient clusters (someone in frame
  for only a couple sampled hits) via a group-size floor. Clustering/tracking design itself not
  yet finalized -- what's below is real benchmarking to size the problem before designing it
  further.

  **Real per-frame benchmark, against 2 real 1080p phone-video fixtures** (`IMG_0760.MOV` 229
  frames/30fps, `VID_20190608_170520884.mp4` 304 frames/29.9fps, both from the real video-samples
  fixture dir), run serially (concurrent runs were tried first and rejected -- ONNXRuntime's own
  CPU intra-op threading already spans many cores per single call, confirmed via `ps aux` showing
  700-1200% CPU for one process alone, so two concurrent benchmark runs just contended for the
  same cores and produced inflated, unusable numbers for both):
  - Baseline (`FaceAnalysis(name='buffalo_l')`, default `.prepare()`): **832-1039ms/frame**.
  - Forcing a real single detector scale (`det_size=(640,640)` explicitly -- default
    `.prepare()` silently runs `DEFAULT_DET_SIZES = [(128,128),(640,640)]`, a built-in
    double-scale pass, not a true single pass, confirmed via `insightface.model_zoo.scrfd`
    source): only **764-974ms/frame**, a small win -- the extra (128,128) pass is cheap since
    it's tiny, so this wasn't where the real cost was.
  - **Real dominant cost, found by inspecting `FaceAnalysis.get()`'s source directly**: it runs
    every loaded non-detection model against every detected face unconditionally --
    `landmark_3d_68` (`1k3d68.onnx`, 143MB) and `landmark_2d_106` (`2d106det.onnx`, 5MB) included.
    A full-codebase grep confirmed **neither landmark output is read anywhere in this repo, on
    the image side either** -- pure wasted compute inherited from `FaceAnalysis(name='buffalo_l')`
    being constructed with no `allowed_modules` filter. Left alone on the image side (stable,
    not touched per the user's explicit instruction) but no reason to carry into new video code.
  - **Lightweight combo, real per-frame timing: 263-344ms/frame (~3x the baseline)** --
    `allowed_modules=['detection','recognition','genderage']` (drops both landmark models) AND
    swaps the detector for `buffalo_s`'s `det_500m.onnx` (2.5MB SCRFD-500MF, downloaded and
    verified working) in place of `buffalo_l`'s `det_10g.onnx` (16.9MB SCRFD-10GF) -- same
    `Face` object interface (bbox/kps/det_score) so it drops straight into the same recognition
    call unmodified. **Recognition stays `w600k_r50.onnx` (buffalo_l's), never swapped** --
    verified the resulting embedding is still real 512-d output (`shape=(512,), norm=18.33`),
    numerically the same model as the image pipeline's own encodings, so video faces will be
    directly comparable to existing `Person` galleries with no separate embedding space. The
    lighter detector does find fewer faces per frame than `det_10g` (expected, lower-capacity
    model) -- an explicit, accepted tradeoff per the user ("we want the encodings to be
    consistent... but the detector can be less capable").
  - **No native batch-inference path exists** in this installed insightface version (1.0.1) --
    `SCRFD.detect()` is single-image only, no multi-frame tensor dimension. Not pursued: CPU
    inference already spans many cores per single call (same reasoning behind this session's own
    earlier multi-threading revert for image classification -- numpy/ONNX matmul already
    multi-threads internally, so Python-level parallelism just oversubscribes the same cores).
  - **Full-library scope, at the lightweight per-frame cost, against the real backfilled library**
    (237.5 hours total footage, 6,946 videos, ~2min average): sampling every frame is not viable
    (~213 days single-threaded); every 10th frame ~21 days; every 20th frame ~11 days; every 30th
    frame ~7 days. This is a one-time-backfill sizing problem, not an ongoing-cost one -- a new
    video going forward costs only ~55s at a stride-20 sample rate for an average ~2-minute clip,
    same "backlog vs. steady-state" shape as the original image face-extraction backfill.
  - **Resolved: the sequential design.** Rather than "stitch vs. cluster" as alternatives, they're
    sequential stages, resolving a real chicken-and-egg problem raised mid-investigation (bbox+kps-
    only candidates have no embeddings, so pure embedding clustering can't be the first pass):
    **Stage 1** IOU tracking (geometry only, no embeddings) chains raw detections into tracklets
    across consecutive sampled frames -- zero recognition cost spent. **Stage 2** drops
    short/transient tracklets by a length floor before ever paying for recognition -- cheaper than
    filtering after clustering. **Stage 3** encodes only survivors, sparsely (a couple
    representative frames per tracklet, not every frame). **Stage 4** embedding-based stitching
    (with a hard cannot-link constraint) merges tracklets IOU couldn't bridge (occlusion, a cut,
    re-entering frame) and classifies the merged result against the existing `Person` gallery.

  - **DB storage: bbox+kps candidates cost ~15-20x less per row than a full `Face` row.** Measured
    directly against production `face_manager_face` (not estimated): a full row averages **2,125
    bytes** (2,067 of that is `face_encoding_512` alone; `kps`: 61 bytes; box columns: 16 bytes
    combined). A lean candidate table (video FK + frame index + box + kps, no embedding) lands
    around **~100-150 bytes/row**. At stride-20 across the full library (~1.28M sampled frames,
    ~1-1.4 faces/sampled-frame in the real fixtures) that's **~1-2M candidate detections costing
    ~150-300MB temporarily**, vs. 2-4GB if every raw detection stored a full embedding.

  - **Compute: no real "recompute penalty" from the two-pass design.** Pure detection-only
    (`det_500m`, bbox+kps, no recognition/genderage): **37-46ms/frame**. Recognition-only
    re-encode via kps-replay (`rec_model.get(img, Face(kps=...))`, the same trick
    `reencode_missing_faces()` already uses): **~142-144ms/face**. Full detect+recognize+genderage
    together: **279-369ms/frame** -- backing that out, **genderage alone costs ~90-120ms/face**,
    roughly as expensive as recognition itself. **Genderage is dropped entirely for video**:
    `Face.detected_age`/`detected_gender` both have real defaults (`-1`/`"unknown"`), not required
    fields, and this project's own earlier birth-year-cutoff investigation already found real
    doubt about whether `detected_age`'s output is usable at all (a known preschooler's median
    came back 46) -- little is given up by skipping it here that wasn't already questionable for
    images. Net: detect-only (37-46ms/frame, paid once regardless of face count) -> cluster/filter
    -> recognition-only re-encode survivors (142-144ms/face, paid only for faces that matter)
    costs about the same or less total compute than committing to full detect+recognize+genderage
    for every raw detection up front, since the two expensive parts are deferred until you know
    which detections survive.

  - **Two real, confirmed extraction-pipeline bugs found via testing against real fixtures, not
    assumed** (both while re-benchmarking with real formats beyond the two phone videos):
    1. **`cv2.VideoCapture` silently corrupts frames on `.mpg`/`.mts`/`.m2ts` content** -- produces
       all-black frames with no error raised (confirmed: a frame `cv2` returned had `mean=0.0`;
       the identical timestamp extracted via `ffmpeg` gave `mean=121.0` and a real detected face).
       This is exactly why every earlier per-frame timing number in this write-up came from
       `cv2.VideoCapture` and needed correcting -- **confirmed against real production data: ~624
       files (500 `.mpg` + 124 `.mts`/`.m2ts`, ~9% of the library) are at risk.** Fixed by
       switching frame extraction to a single `ffmpeg` subprocess per video, piping raw frames
       (`-f rawvideo -pix_fmt bgr24 pipe:1`) rather than using `cv2.VideoCapture` at all.
    2. **`ffmpeg`'s raw pipe auto-applies the container's rotation side-data**, so a ±90°/270°-
       rotated video's actual output frame dimensions are swapped from `ffprobe`'s raw stream
       width/height. Assuming the un-rotated dimensions when reshaping the raw byte buffer
       produces severely garbled frames (confirmed visually -- a diagonal-shear artifact, not
       recognizable content) and silently zero detections, not an error. Fixed by checking
       `ffprobe`'s `stream_side_data` rotation and swapping width/height before reshaping whenever
       rotation is ±90/270. Re-verified against the real `IMG_0760.MOV` fixture (rotation=-90):
       correctly recovers a normal, upright real photo and all 10 real faces once fixed.
    3. **Black-frame sanity check, built and validated**: probe the first ~10 sampled frames per
       video, flag the video if >=50% come back near-zero mean pixel value (`< 1.0`). All 6 real
       fixtures passed clean after the two fixes above -- this is the guard that would have caught
       the original `cv2` corruption immediately (as "video flagged, needs investigation") instead
       of silently reporting "no faces found." Not yet wired into the real ingestion pipeline --
       validated only in the investigation scripts so far.
    4. **Corrected detection-only cost, post-fix, across 6 real diverse videos** (phone MOV/MP4,
       AVCHD MTS/M2TS, old MPG, WMV): **38-82ms/frame** -- consistent with, not wildly different
       from, the earlier (bugged) `cv2` numbers, but now trustworthy.

  - **Track fragmentation is real and substantial with plain IOU tracking alone** (0.67s between
    samples, boxes grown 25% before computing IOU, `iou_thresh=0.3`) -- most tracks are singletons;
    only a minority survive length>=2. Real per-video breakdown (tracks / % surviving >=2 frames):
    `IMG_0760.MOV` 3 tracks/33%, `VID_2019...mp4` 14/14%, `00023.MTS` 32/56%, `20191005...m2ts`
    12/33%, `20120309...mpg` 14/29%, `Bear Hands.wmv` 9/67% -- directly motivating Stage 4 as
    necessary, not optional.

  - **Levers tried, in order of actual value per unit cost (parameter sweep + two follow-up
    experiments, all against real fixtures):**
    - *Lowering the IOU threshold* (0.3->0.2->0.1) and *growing boxes more before matching*
      (25%->50%) each help a little, unevenly by content, with diminishing/overlapping returns
      when combined (e.g. on `00023.MTS`, `iou>=0.3,grow=50%` ~= `iou>=0.2,grow=25%`; at
      `iou>=0.1` further growing boxes made zero additional difference). Neither is a big lever on
      its own.
    - *Sampling twice as often* (0.67s -> 0.33s between samples) gives a real but more modest
      continuity win than raw sample-count numbers suggest -- **track length in sample-count
      trivially doubles with 2x the sample rate regardless of any real improvement** (a units
      artifact, correctly caught mid-investigation), so the honest metric is real TIME SPAN of the
      longest track: `00023.MTS` 4.67s->8.68s (1.86x), `VID_2019mp4` 1.34s->1.68s (1.25x) -- real,
      but content-dependent and smaller than it first looked. Cost: detection-only time scales
      close to the expected ~2x (confirmed by isolating decode-only vs. detect-only wall time:
      decode is a large FIXED cost since the current extraction decodes every frame in the video
      regardless of stride, e.g. `00023.MTS` decode-only stayed ~24-26s at both sample rates while
      detect-only genuinely went 4.41s->10.56s) -- an earlier claim that "total cost barely
      increased" was an artifact of decode dominating this specific unoptimized implementation,
      not a real property of sampling rate; a version that skips decoding unsampled frames
      entirely (e.g. `ffmpeg`'s own `-vf select=...` frame-selection filter) would scale total
      cost close to 2x with 2x the sample rate, same as detection does here. Not yet built.
    - **Gap tolerance -- the best lever found, and free.** Allowing a track to survive up to K
      consecutive missed sampled frames before closing (matching resumes against the last known
      box) improves real continuity at ZERO extra detection cost (same samples, smarter linking
      only). `00023.MTS` max real span: 4.67s (K=0) -> **9.34s (K=1)** -> 8.01s (K=2) -- note this
      is **not monotonic**: more tolerance doesn't always help further, since the optimal
      bipartite matching can resolve a different pairing as tolerance loosens, shifting which
      segments link rather than only ever adding more. `VID_2019mp4`: length in *samples* stayed
      flat at 3 across K=0/1/2 while real span still climbed 1.34s->2.01s->3.35s, confirming gap
      tolerance bridges real time between existing detections rather than manufacturing new ones.
      Recommended as the first lever to reach for, ahead of sampling rate or IOU/box-growth
      tuning.

  - **Stage 4 (embedding stitching) prototyped and visually validated against a real fixture.**
    Design: for each surviving tracklet, encode a representative embedding at its start and end
    (recognition-only kps-replay); build a cost matrix of end-to-start cosine similarity between
    every ordered pair of tracklets, with a **hard cannot-link constraint** -- any pair that
    overlaps in time (or isn't in the right order) gets an disallowed cost, never eligible to
    merge, regardless of embedding similarity (ground truth: the same person can't be in two
    places at once) -- then a single `linear_sum_assignment` call resolves the optimal set of
    stitches at once, thresholded at cosine similarity `>=0.4`. Union-find merges chains of
    pairwise stitches into final groups. **Real result against `00023.MTS`'s gap=1 tracks: 26 raw
    tracks -> 19 groups.** The biggest merge chained 5 separate tracks across gaps up to 9.34s
    into one continuous **21.35-second span** (more than 2x the best result gap-tolerant IOU
    tracking alone achieved). **Visually confirmed correct**, not just numerically plausible --
    pulled and inspected face crops from all 5 segments (same glasses, same hair, same reclined
    profile pose throughout) -- this project's own established discipline (the `.ignore`-bucket
    clustering investigation found numerically-clean clusters that were visually incoherent) made
    this check necessary, not a formality, and here it held up.

  - **Full-track (m*n) matching instead of start/end-only representatives -- investigated
    2026-09-07, real improvement confirmed, but surfaced a deeper structural problem with
    bipartite matching itself (see below).** Motivated by a direct question: rather than encoding
    only each tracklet's first/last face, encode EVERY face in a tracklet and compare all m*n
    face pairs between two tracklets, using an aggregate (max/mean/median/percentile) as the
    tracklet-to-tracklet score. This is the same question this project already answered for
    `classify_unassigned()` (max/1-NN found "maximally vulnerable to any single noisy face,"
    percentile-gating held up better) -- worth re-testing here since track sizes (2-8 faces) are
    much smaller than a `Person` gallery (hundreds+), where that logic was established.
    - **Real result**: all four aggregates (max/mean/median/p75) agreed on one clearly valuable
      stitch the start/end-only baseline missed entirely -- confirms full m*n comparison is a
      real improvement, not just start/end-only being "good enough."
    - **But max-similarity specifically pulled in 2 more borderline stitches that mean/median
      rejected** -- visually checked via saved face crops (this project's own established
      discipline: numerically-clean-looking merges have been visually wrong before, e.g. the
      `.ignore`-bucket clustering investigation). Both disputed pairs were infant faces --
      plausibly the same baby, genuinely hard to call even by eye, exactly the low-information
      case this project's outlier-vulnerability concern was originally about.
    - **p90 did NOT behave as an intermediate safety margin between mean and max, as hoped** --
      it produced an IDENTICAL result to pure max (same 16 groups, accepted both disputed
      pairs). Root cause: with only 2-8 faces per track, the 90th percentile of a small m*n
      matrix often collapses to (or very near) the actual max value -- there simply aren't enough
      data points to spread a percentile meaningfully away from the extreme. The "percentile
      protects against outliers" logic that worked for `classify_unassigned()` assumed
      hundreds-to-thousands of gallery faces; it doesn't transfer cleanly to comparing two small
      tracks. A genuinely intermediate option (e.g. top-k average with a small fixed k, or
      requiring >=2 pairs above threshold) was identified but not yet tried.

  - **Bipartite (Hungarian) matching for stitching is the wrong tool structurally, not just
    mistuned -- root-caused via a real diagnostic, not assumed.** Started from a concrete
    observation: the same visually-identical person (glasses, reclined pose) appeared split
    across 4 separate final groups in the contact-sheet review (real crops pulled and inspected,
    not just counts) that should have chained into one identity. Dumped the actual similarity
    values for the specific missed boundary pairs: **0.759 and 0.658 -- both comfortably above
    the 0.40 threshold**, so this was never a threshold problem. `linear_sum_assignment` forces a
    COMPLETE permutation across all n tracks (every row assigned to some column); with mostly-
    disallowed (1e6-cost) entries and only a sparse few real candidates, the solver can be forced
    to sacrifice a genuinely good pair to avoid an even-worse forced pairing elsewhere in the
    matrix -- a real, structural failure mode of naive Hungarian matching on a mostly-sparse cost
    matrix, not a parameter to tune away.
    - **Tried: dummy "stay unmatched" padding** (standard optional-assignment trick -- pad the
      cost matrix with a same-cost "remain unmatched" option per track, at exactly the
      similarity threshold, so the solver is never forced to accept worse-than-threshold). First
      attempt had a real bug (only zeroed the DIAGONAL of the dummy-to-dummy block instead of the
      whole block, which incorrectly re-introduced the same forced-sacrifice problem via a
      different path -- caught by testing, produced 0 stitches at all, obviously wrong). Once
      fixed, it recovered more real matches, but surfaced a NEW, different problem:
    - **Leapfrogging**: with "stay unmatched" available, the solver is free to pick the
      objectively-highest-raw-similarity pair ANYWHERE in the video, with zero preference for
      temporal proximity -- produced nonsensical long-distance stitches (e.g. a track near the
      very start matched to one near the very end, skipping right over a much closer, more
      sensible candidate in between). Confirmed this isn't an assignment-strategy artifact:
      **iterative re-solving** (repeatedly solve, accept threshold-passing stitches, remove those
      tracks, repeat) reproduced the exact same 16-group result as one-shot matching and still
      missed the known-good 0.759 link -- traced to a real bug of its own (conflating a track's
      sender-role and receiver-role as a single "matched, remove entirely" state, when a mid-chain
      track legitimately needs both an incoming AND an outgoing link simultaneously; fixed by
      tracking sender/receiver availability separately) -- and even fixed, round 2 still suffers
      the SAME forced-full-permutation problem on whatever's left, just on a smaller pool, so
      iterating doesn't structurally fix anything. **Pure greedy** (repeatedly take the single
      best remaining candidate pair, no assignment-forcing at all) was tried as the simplest
      alternative and made leapfrogging WORSE, not better (a track at the very start of the clip
      matched to one at the very end, sim=0.866) -- confirming the missing ingredient across
      every variant tried is the same: none of them have any notion of temporal proximity.
    - **Tried: restricting candidates to a maximum real-time gap, plus a threshold sweep**
      (10s/15s/unlimited cutoffs x 0.40/0.35/0.30 thresholds, 9 combos on the plain one-shot
      solver). Gap cutoff had ZERO effect in any combo (the real max useful gap in this fixture
      is 9.34s, safely under every cutoff tested, so nothing was ever actually excluded) --
      confirming the missing 0.759 link is NOT explained by gap size at all. Lowering the
      threshold only added one new, unrelated low-confidence link; the known-good link stayed
      missing across all 9 combos, confirming (again) the failure is the forced-permutation
      stealing problem, not a threshold or gap issue.
    - **Tried: gap-restriction combined with dummy-padding together** (the two independently-
      diagnosed fixes, combined). Did not converge to a clean result either -- produced MORE
      groups than the plain one-shot baseline (21 and 20 vs. 16), and the leapfrogging problem
      reappeared at the 15s cutoff (a start-of-clip track matched to one 14.68s later). Diminishing
      returns from continuing to patch the bipartite-assignment formulation with more constraints.
    - **Conclusion, and the actual next step**: the real mismatch is structural, not a tuning gap
      -- bipartite assignment limits every track to at most ONE link in and ONE link out, but the
      real problem (one person reappearing 3+ separate times across a video, with OTHER people's
      tracks interleaved between appearances) is inherently a **many-to-one grouping problem**,
      which no combination of gap cutoffs, dummy padding, iteration, or greedy selection can fix
      by construction. The project already has the right tool for this shape of problem:
      `verification_clustering.py`'s complete-linkage clustering (used for `Face.
      verification_cluster_group`), which naturally handles "one identity, many separate
      appearances" without any one-in/one-out limitation. Plan: apply that directly to
      tracklet-level representative embeddings, enforcing the cannot-link constraint by forcing
      temporally-overlapping tracklet pairs to an infinite/disallowed distance before clustering,
      rather than continuing to force this into an end-to-start bipartite matching formulation.

  - **Clustering tried (2026-09-07): confirms the problem is NOT the matching algorithm --
    every method tried, bipartite or clustering, fails to consolidate this video's known-hard
    case, pointing at embedding variability itself as the real limiter, not algorithm choice.**
    Built `AgglomerativeClustering(linkage='complete', metric='precomputed')` over EVERY
    individual face embedding across all 26 tracklets (not track-level representatives), with
    cannot-link enforced by forcing temporally-overlapping cross-track pairs to a disallowed
    (1e6) distance in the precomputed matrix. **First run found a new bug before ever reaching
    the real question**: 15 of 26 tracks got split across multiple cluster labels internally,
    despite every face within one track being same-person by construction (that's what the IOU
    tracker guarantees) -- complete linkage's strict worst-pair-must-clear-threshold rule was
    rejecting even a track's own natural pose/lighting variation, since intra-track pairs had
    been left at raw computed distance instead of being forced together. **Fixed with a
    must-link constraint** (force `dist=0` for same-track pairs, alongside the existing
    cannot-link forcing for overlapping cross-track pairs) -- re-run confirmed 0 internally-split
    tracks, so the must-link fix is necessary and sufficient for internal consistency.
    - **But the corrected result still didn't consolidate the known case**: 26 tracks -> 21
      groups at the project's own default `cos_threshold=0.6`, and a full sweep down to a very
      loose 0.35 (`AgglomerativeClustering` distance threshold up to 1.14, well past any value
      this project has used elsewhere) **never merged the four scattered appearances of one
      real person (a glasses-wearing woman) together** -- they stayed as separate groups at
      every single threshold tested, while OTHER unrelated tracks increasingly blobbed together
      as the threshold loosened (a 6-track group appeared by cos=0.55) -- the classic
      single-criterion tradeoff (loosen enough to catch the real merge you want, and you catch
      wrong merges elsewhere first). Root cause: complete linkage requires the WORST pairwise
      distance across every frame in both tracks to clear threshold; with all-frames-per-track
      (not just start/end reps), one bad frame (motion blur, extreme angle, partial occlusion) in
      either track is enough to permanently block the merge, no matter how good every other frame
      pair looks. This is the same "worst-case sensitivity to outlier frames" problem the earlier
      max-vs-mean/median m×n investigation already flagged, just showing up from the opposite
      direction here (too conservative for genuine merges, instead of too lenient for disputed
      ones).
    - **Tried average linkage instead (same must-link/cannot-link-constrained precomputed
      matrix)**, since averaging dilutes a single bad-frame outlier rather than being blocked by
      it outright. Recovered more of the target merges as the threshold loosened (2 of 4 by
      cos=0.55, one further one folded into a same-labeled 6-track group by cos=0.45) -- but
      **visual verification via a contact sheet (representative middle-frame crop per track,
      grouped by final cluster label) showed the same person was STILL split across 3 separate
      groups at cos=0.5** (groups of tracks `(0,120)+(880,880)+(1000,1080)+(2040,2280)+(3200,3320)`
      vs. `(160,280)+(640,960)+(1120,1120)+(1400,1440)+(2840,2840)` vs. the lone `(3080,3160)` --
      all three visually confirmed as the identical woman, same glasses, same setting) -- despite
      average linkage's known risk of over-merging *different* people once loosened far enough to
      catch this (partially already visible: cos<=0.45 started folding unrelated tracks into a
      shared 6-track group). Per this project's own established discipline (numerically-clean
      groupings have been visually wrong before, and the reverse -- numerically-plausible
      non-merges hiding a real same-person split -- is just as much a trap), this was checked by
      eye, not assumed from group counts alone.
    - **Conclusion: this is not a linkage-criterion or threshold-tuning problem at all.** Every
      approach tried across this entire investigation -- one-shot bipartite matching, dummy-padded
      optional assignment, iterative re-solving, pure greedy, gap-restricted variants, complete-
      linkage clustering, and average-linkage clustering, at every threshold from strict to very
      loose -- fails to fully consolidate this one real, hard case (a person whose appearances are
      scattered non-adjacently through a clip with long gaps and highly variable pose/lighting
      between them). The common thread is that ALL of these methods reduce a track's identity down
      to a single aggregate similarity number (max, mean, a worst-case, an average) compared
      against a single global threshold -- and for a person with this much real per-frame
      variability, no single number/threshold combination cleanly separates "same person, just a
      bad frame" from "different person, coincidentally similar." Fixing this for real would need
      either better/more consistent per-frame embeddings (not a stitching-algorithm change at all)
      or genuinely new signal beyond raw embedding similarity (e.g. the co-occurrence/temporal-
      context priors already brainstormed-but-unscoped for the image pipeline's own outlier-
      rejection work, or a two-stage human-assisted merge suggestion rather than a fully automatic
      one). **This validates, rather than overturns, the project's original Phase 3 MVP decision**
      (see this section's own opening paragraph): accept near-duplicate/split-identity entries per
      person per clip as a known limitation for the first shippable version, leaning on the
      existing `verification_cluster_group` review tooling for human cleanup, rather than blocking
      Phase 3 on solving fully-automatic cross-gap stitching -- which this investigation now shows
      is a substantially harder, open-ended problem than originally scoped, not a matter of
      picking the right algorithm or threshold. Real cross-frame tracking beyond simple IOU+gap-
      tolerance remains explicitly Phase 3.5, not Phase 3, per the project's existing phasing.

  - **Real, confirmed bug found and root-caused (2026-09-07): raw-pipe frame extraction never
    deinterlaces, producing visible combing/blocking artifacts on any moving subject in an
    interlaced source video -- this, not a transition, is what the user actually spotted as
    "encoding artifacts" on a contact sheet.** Started from the user's own correction ("It's not
    a transition, just some faces on the contact sheet had some encoding artifacts, some
    interleaving I think") after an initial hypothesis (a same-location-neighbor-frame check to
    catch spurious detections from editing transitions) came back clean against two real
    fixtures with no editing cuts, so didn't actually test the real concern. Checked
    `ffprobe`'s `field_order` directly against all 8 real fixtures: **`00023.MTS`,
    `20120309182349.mpg`, and `20191005094939.m2ts` are all `field_order=tt`** (top-field-first
    interlaced) -- exactly the same three formats already flagged elsewhere in this file as
    needing the `ffmpeg` raw-pipe fix for `cv2`'s frame-corruption bug (mpg/mts/m2ts, ~624 real
    files, ~9% of the library). The other 5 sample formats (wmv/mov/avi/3gp/mp4) came back
    `unknown` or `progressive`. **Visually confirmed via a real face crop**: extracted the same
    frame from `00023.MTS` twice, once via the current raw-pipe command
    (`ffmpeg -i ... -f rawvideo -pix_fmt bgr24 pipe:1`, no deinterlace) and once with
    `-vf yadif=0` inserted -- the raw version shows clear blocky/combed distortion on a moving
    subject at the frame's edge (a baby's face, genuinely in motion at that instant), while the
    deinterlaced version is visibly clean at the same crop. The neighbor-frame consistency check
    originally built to investigate transitions is a real but separate, complementary safeguard
    (it would catch severely corrupted per-detection embeddings after the fact); it does not fix
    the actual root cause here, which is upstream in extraction itself.
    - **Fix, not yet applied to any real pipeline code** (this remains investigation-only, same
      as the rest of Phase 3): insert `-vf yadif=0` (or `bwdif=0`, generally regarded as
      somewhat higher quality, at the same call-site cost) into the `ffmpeg` raw-pipe command
      whenever the source's `ffprobe`-reported `field_order` isn't `progressive`/absent --
      gate on that check rather than deinterlacing unconditionally, since running a deinterlace
      filter on already-progressive content can still alter/blend frames unnecessarily. `yadif`'s
      default mode (`mode=0`, `send_frame`) outputs one deinterlaced frame per input frame
      (not one per field), so frame indices/strides/timestamps used elsewhere in this pipeline
      (IOU tracking, sample stride math) stay unaffected by turning this on.
    - **Checked: deinterlacing also recovers real missed detections, not just cleaner embeddings
      on already-found faces -- confirmed on the fixture with the most motion/faces.** Ran the
      lightweight `det_500m` detector across every sampled frame of all 3 confirmed-interlaced
      fixtures, once raw and once with `-vf yadif=0`. `00023.MTS` (by far the busiest fixture,
      the same one used throughout this whole tracking/stitching investigation): **85 -> 94 total
      detections across the same 84 sampled frames (+9, +10.6%)** with deinterlacing -- a real,
      meaningful recall gain, consistent with the visual finding above (a combed/corrupted frame
      can plausibly hide a face from the detector entirely, not just degrade its embedding once
      found). The other two fixtures (`20120309182349.mpg`: 19->18; `20191005094939.m2ts`:
      17->18) showed only noise-level ±1 differences -- not a meaningful signal either way, but
      also both have far fewer sampled frames (6 and 42) than `00023.MTS`'s 84, so this may just
      be too small a sample to show the same effect, not evidence the effect is absent there.
      **Not yet done**: visually spot-checking the 9 newly-recovered `00023.MTS` detections to
      confirm they're genuine faces and not false positives introduced by whatever `yadif` does
      to frame content in some other way -- worth doing before treating this recall number as
      final, though the combing-artifact mechanism already visually confirmed above makes a real
      recall gain the more likely explanation than new false positives.
    - **Re-ran the full clustering association (both linkage types, same threshold sweep, same
      must-link/cannot-link-constrained matrix) on `00023.MTS` with `-vf yadif=0` applied, to see
      whether cleaner embeddings fix the earlier consolidation failure -- real improvement, but
      NOT a full fix.** 31 tracks now (up from 26 pre-deinterlace, consistent with the +9
      recovered detections). **Complete linkage at the project's own default `cos_threshold=0.6`
      now produces a real 6-track merged group**
      (`(0,120)+(880,880)+(1000,1080)+(1960,1960)+(2040,2240)+(3200,3320)`) where before
      deinterlacing the same threshold left these as isolated singletons/small pairs -- a
      genuine, meaningful consolidation gain from the cleaner embeddings alone, with zero
      threshold changes. **But it's still not the full answer**: a contact-sheet visual check
      (same discipline as before) showed the SAME real person (a glasses-wearing woman) still
      split across **3 separate groups** even at the more lenient average-linkage/cos=0.5 pass --
      the 6-track group above, a second 5-track group
      (`(160,360)+(640,960)+(1960,1960)+(2480,2480)+(2840,2880)`) that's visually identical to
      the first, and a lone singleton (`(3080,3160)`), also visually her. Complete linkage at
      cos=0.6 only ever recovered the first cluster -- the second cluster's tracks stayed
      unmerged singletons at that threshold, not merged with the main group or each other.
      **Conclusion: deinterlacing is a real, worth-doing fix in its own right (recovers missed
      detections, produces cleaner embeddings, measurably improves consolidation at the
      project's own default threshold with zero other changes) -- but it does not resolve the
      deeper track-stitching problem already concluded above.** The same person can still
      fragment across multiple visually-identical groups even with clean, deinterlaced input,
      confirming the earlier conclusion (this is a real per-appearance embedding-variability
      limit -- pose, expression, lighting across genuinely different moments in the clip -- not
      an artifact of any single upstream data-quality bug). Both fixes belong in the eventual
      real pipeline (deinterlacing for its own clear, independent benefit; the MVP acceptance of
      split-identity entries for the deeper problem it doesn't solve) -- neither supersedes the
      other.

  - **Ideas 1-3 tried together (2026-09-07), plus a 4th (the user's own suggestion) that
    reframes the whole problem -- real, meaningful progress on the first three, and the 4th is
    a genuinely different, more promising angle than continuing to chase track-to-track
    stitching.** All four tested together against the same deinterlaced `00023.MTS`:
    1. **Re-detect/re-encode each track's representative frame(s) with the full `buffalo_l`
       `det_10g` detector** (already loaded anyway, as part of the recognition `FaceAnalysis`
       app) instead of only ever using the lightweight `det_500m` detections from the bulk
       per-frame sampling pass -- cheap, since only ~2 frames per track (not every sampled
       frame) need the heavier pass.
    2. **Pick each track's best 2 frames by a cheap quality score** (`det_score * box_area`)
       instead of feeding every frame (which is what fed complete linkage's worst-pair
       sensitivity) or an arbitrary middle frame into the distance matrix.
    3. **Added a clothing/torso color-histogram (HSV) as an auxiliary signal**, combined with
       the face-embedding distance.
    4. **The user's own idea**: classify each track's representative embedding against the REAL
       production `Person` gallery (the actual `classify_unassigned()` machinery, not a
       reimplementation) to see whether tracks that don't merge with each other inside the video
       independently converge on the same already-confirmed real person.

    **Ideas 1+2 together gave real, measurable progress**: at complete-linkage cos=0.5, the
    previously-unmergeable `(3080,3160)` singleton (isolated across every method tried all
    session, including after deinterlacing alone) now merges into a real 7-track group. A
    contact-sheet visual check confirmed the glasses-woman is now split across only **2** groups
    (down from 3-4 before) -- real progress, though still not full consolidation. Crop quality
    is visibly sharper than every earlier contact sheet in this investigation, consistent with
    `det_10g` producing better boxes/kps than `det_500m` on these specific frames.

    **Idea 3 (clothing histogram) did not help, and made things measurably worse**: at the same
    thresholds, adding the naive HSV-histogram distance term produced MORE groups, not fewer
    (17 vs. 11 at average-linkage/cos=0.5) -- the crude implementation (a fixed-height crop
    below the face box) likely picks up inconsistent background/blanket/lighting per frame
    rather than a stable clothing signal, adding noise rather than a genuine second independent
    signal. Not pursued further in this form; a real implementation would need a proper
    person/torso detector, not a fixed offset below the face box.

    **Idea 4 is the standout result of this whole investigation, and reframes the problem
    rather than incrementally improving on it.** Saved each of the 31 tracks' representative
    embeddings (averaged over their best-2-frame reps, computed with `det_10g`) and classified
    them, read-only, against the REAL live production gallery (440 likely people, 281,971 real
    gallery faces) using the exact same `sim_99th`-vs-gallery-size-bucket-threshold gate
    `classify_unassigned()` already uses (called only `faceAssigner.__init__()`/
    `load_encodings()`/`_build_concatenated_gallery()` directly -- never `classify_unassigned()`
    itself or any `.save()`, so this stayed strictly read-only against production; embeddings
    were transferred between the dev/test container and the live `picasa_api` container via
    base64 over `docker exec`, since `docker cp` writes through a filesystem view this session's
    sandboxed shell can't see -- see the workaround noted where first hit). **Every one of the
    13 tracks visually confirmed as the glasses-wearing woman -- spread across the entire
    clip, never fully consolidated by any clustering method tried all session -- independently
    classified to the SAME real person, "Jessica Lewis," with strong, unambiguous confidence
    (similarity 0.5-0.63, comfortably clear of the 0.394 threshold, zero competing
    candidates)**. The baby's tracks, by contrast, were genuinely ambiguous -- bouncing between
    3 close candidates ("Gwendolyn Lewis"/"Nathaniel Lewis"/"Liam Lewis," scores within
    0.02-0.03 of each other) and even pulling a different family's children as the closest
    match on a couple of tracks -- consistent with this project's already-known difficulty
    classifying young children (siblings look alike as infants), not a new video-specific gap.
    **Conclusion: rather than continuing to invest in fully solving in-video track-to-track
    stitching, classifying each track directly against the existing real-person galleries
    (built from years of real photos, with far more appearance-variability coverage than one
    video clip alone could ever provide) already solves the consolidation problem outright for
    anyone well-represented in the gallery -- sidestepping the track-stitching problem entirely
    rather than solving it.** The remaining difficulty (the baby's ambiguous 3-way split) is the
    SAME already-known gallery-classification limitation the image pipeline already has for
    young children, not a new problem introduced by video. This suggests Phase 3's real design
    should lean on gallery classification as the primary consolidation mechanism, with
    in-video track/cluster stitching kept only as a cheap first pass (to avoid re-classifying
    the same track's every single frame) rather than as the mechanism relied on for full
    identity consolidation.

  - **Re-ran the same ideas-1-3 pipeline at 2x sample density (2026-09-07), per the user's own
    request -- the best in-video consolidation result of the entire investigation.** Halved the
    stride (`fps * 20/30 / 2`, giving 50 tracks instead of 31) with everything else unchanged
    (deinterlaced, det_10g re-detect on best-2-frame reps, same threshold sweep). At
    `face+clothing`/average-linkage/cos=0.5, **a single 17-track group now consolidates nearly
    all of the glasses-woman's appearances** (`0-120, 140-200, 280-420, 640-820, 1060, 1120,
    1540, 1700, 1960, 2480, 2540, 2820-2900, 2980, 3080-3160, 3200-3340`) -- visually confirmed
    clean via a contact sheet, no contamination in the sampled crops. This is a real step
    change from every earlier attempt this session (which topped out at 2-3 separate groups for
    her, even combining every other idea tried) -- denser sampling gives IOU tracking more
    continuity to work with, directly reducing the frame-to-frame gaps that were the root cause
    of fragmentation. (One baby-track group's crops showed a possible partial contamination --
    one crop in a 6-track baby group had what looked like a stray glasses edge -- not fully
    confirmed, worth a closer look if this stride is adopted.)
    - **Gallery cross-check reconfirmed the same result independently**: re-ran the read-only
      production classification (same method as before) against all 50 of the new tracks --
      virtually every track in the new 17-track group matches "Jessica Lewis" with strong
      confidence (0.5-0.63 similarity), the same person identified before, now with even more
      redundant confirmation across more, denser track samples. **Also surfaced a new, distinct
      detail the lower sample density had missed**: one track (span `2340,2340`) confidently
      matches a THIRD real person, "Emma" (sim99=0.576 vs. threshold=0.558, a small 19-face
      gallery) -- a genuinely different person appearing briefly in this clip that hadn't been
      flagged in the 1x-density run at all. This is itself a point in favor of gallery
      cross-checking over pure in-video stitching: a person who appears only once, briefly, and
      isn't visually similar to anyone else in the clip has no chance of being "stitched" to
      anything internally, but is still directly and correctly identifiable against the existing
      gallery from a single track alone.
    - **Practical cost note**: 2x sampling means 2x the detection cost for the same video length
      (per the earlier stride-cost analysis in this file) -- worth weighing against the real
      consolidation gain when finalizing Phase 3's shipped stride, not treated as a free win.

  - **Follow-up (2026-09-07): root-caused the baby's specific fragmentation directly, and tested
    (and rejected) a targeted local-recrop recovery idea.** Prompted by the user's own question
    ("was it just that it wasn't detected sometimes?"). Reconstructed the full per-sampled-frame
    detection log (not just track spans) at 2x stride and checked every track-to-track boundary
    directly against it. Confirmed a real mix of causes, correcting an oversimplification from
    the prior answer: some apparent "baby gaps" are actually just a DIFFERENT concurrent person
    (the woman) being detected in between, not a missed baby link at all (e.g. the single-frame
    track at sample 40, a small 31,803px box, is followed by 4 frames of a much larger,
    differently-positioned box that matches the woman's own track almost exactly) -- track spans
    alone conflate this, real per-frame detection logs don't. But genuine multi-frame detector
    dropouts are also real and common: 20 sampled frames across the video have ZERO detections
    at all (not a linking failure, an outright miss), including 4-in-a-row (samples 1320-1380,
    ~1.3 real seconds) and 3-in-a-row (samples 2260-2300) stretches.
    - **Tested the user's proposed fix: re-run detection on just a ~3x-larger crop (100% padding
      each side, "200% bigger") centered on the track's last known box, for every one of those 20
      zero-detection frames** -- testing whether the misses were a resolution/scale problem
      (a face too small relative to the full 1920x1080 frame for `det_500m` to find, but
      recoverable once effectively zoomed in, since the crop still gets resized up to the
      detector's own 640x640 input). **Result: only 1 of 20 recovered a face.** Not a resolution
      problem for the other 19.
    - **Visually confirmed why, via 4 representative crops**: a genuine, complete absence (the
      baby had moved entirely out of that spatial region -- one crop shows only a crib mobile and
      blinds, another only blinds, no face anywhere in frame); a real face turned fully away from
      camera (the back of the head, visible hair only -- no 2D detector, at any resolution, finds
      a face with zero facial features presented); and one edge case that was actually a
      different real crop-boundary issue (the WOMAN partially cut off at the crop's edge, not a
      baby miss at all). **None of these are fixable by searching harder in the same spatial
      region at higher effective resolution** -- the information a face detector needs (a
      forward-facing face) simply isn't present in that frame, at that location, for that
      reason.
    - **Follow-up (same day): tested the user's homography idea too -- a real, but ultimately
      misleading, improvement that a closer visual check corrected.** If the camera itself
      panned/shook (common in handheld home video), the subject could still be genuinely in
      frame, just spatially shifted -- a fixed-location search would miss it even if perfectly
      detectable. Estimated a global frame-to-frame homography (ORB features + RANSAC, typically
      100-800 inlier matches) between each track's last real frame and its gap frame, warped the
      last known box through it to get a motion-corrected predicted location, and searched there
      instead. **Raw result looked like a real win: 3/20 recovered vs. the static crop's 1/20 --
      a 3x improvement.** But visually inspecting all 3 "recoveries" (drawing the detected box on
      the actual crop) told a different story: **all three are the WOMAN's face, not the baby's**
      -- in every one, the baby's own head is visible in the same crop, genuinely turned away
      from the camera, still completely undetected. The apparent improvement was an artifact of
      the search region (whether static or motion-shifted) happening to also cover a different,
      easier-to-detect adult face nearby -- not a real baby recovery at all.
      **Corrected conclusion: the true baby-specific recovery rate is 0/20, under BOTH static and
      homography-corrected search.** This is more decisive than the raw numbers first suggested,
      and closes out the "can smarter spatial search recover the baby" question for this
      investigation -- no amount of relocating WHERE to search fixes a frame where the actual
      problem is that no forward-facing baby face exists at all. Hardening detection/tracking
      further has real diminishing returns for a subject that turns away and moves this much
      within a clip -- reinforces (rather than reopens) the standing conclusion above: leaning on
      gallery classification per surviving track, rather than continuing to invest in keeping a
      fast-moving/frequently-turned-away subject continuously tracked, is the more promising
      direction for Phase 3.

  - **Follow-up (same day): tested the user's optical-flow/tracker idea too -- a genuine visual
    tracker (not just re-detection) that follows the object itself frame-by-frame, independent
    of whether a face is oriented toward the camera. This produced the THIRD consecutive
    "apparent recovery, actually the wrong person" result, now a clear pattern worth stating as
    its own conclusion.** `cv2`'s CSRT/KCF/MedianFlow trackers need opencv-contrib, not present
    in this environment's build (only `TrackerMIL`, plus three deep-learning trackers needing
    external model files not downloaded) -- used `TrackerMIL`, seeded on each track's last known
    box, updated through EVERY raw (undecimated) frame across the gap (not just the sparse
    stride samples), checked against the real next detection's box via IOU at the far end.
    - **First pass (pure open-loop tracking, no mid-gap correction) got 2/14 gaps bridged
      (IOU>0.3)** -- a real, nontrivial-looking result. The user's own follow-up question caught
      a real gap in the test before it was over-interpreted: **"did we reinitiate when we lost
      the track then found a face again?"** -- correctly identifying that the test as originally
      built never checked for or used a fresh detection mid-gap to correct drift; it was pure
      open-loop tracking from one endpoint to the other with no correction opportunity, which
      would especially hurt the longest gaps (up to 160 raw frames, ~2.7s, with zero real anchor
      point in between).
    - **Visually checking the 2 "successes" first, before building the more complex
      reinitialize-on-reacquire version, immediately explained why they succeeded: both are
      the WOMAN's face, not the baby's** -- one crop shows the baby's head clearly present but
      turned away, completely untracked; the other shows only the baby's arm, no head visible at
      all. The tracker had drifted onto (or was tracking, from the start, a box that actually
      included/favored) the adjacent, larger, more consistently front-facing adult, not the
      harder subject the gap was nominally about. **True baby-specific bridging success: 0
      (of however many of the 14 gaps genuinely belong to the baby, which is fewer than 14 once
      the woman's own easier gaps are excluded).**
    - **This is now the THIRD independent method (static re-detection, homography-corrected
      re-detection, and continuous appearance tracking) where an apparent "recovery" turned out,
      on visual inspection, to be the wrong person entirely** -- not three separate weak
      results, but one consistent pattern: whenever a search region is widened, relocated, or
      tracked forward in time near where the baby was last seen, it reliably catches the nearby
      easier adult face instead, because she is more often correctly-oriented and more visually
      stable than the baby is at exactly the moments the baby's own detection fails. **This
      strongly reinforces (with three-for-three consistency, not once) that the baby's detection
      gaps are not a search-strategy problem at all** -- no combination of WHERE to look (static,
      motion-compensated, or continuously tracked) recovers her specifically, because the
      recoverable signal in these frames belongs to a different person. The reinitialize-on-
      reacquire refinement was not built, since the premise it would test (mid-gap correction
      improving on open-loop tracking) is moot if the tracker's target was never reliably the
      baby to begin with in the cases that "worked."
    - **This closes out the detection/tracking-hardening line of investigation on firmer
      footing than either of the two prior attempts alone** -- gallery classification per
      surviving track remains the clear, validated direction for Phase 3, and further engineering
      effort on video-side detection/tracking robustness for a subject this mobile is not
      expected to pay off, based on three independently-designed, independently-failed attempts.

  - **Not yet decided**: final sample stride and gap-tolerance value to actually ship with, the
    group-size floor threshold for dropping transient tracklets, the full-track aggregation
    metric (leaning toward mean/median over max, given the outlier-vulnerability findings above,
    but not finalized), and whether to also pursue the ffmpeg-side frame-selection optimization
    (skip decoding unsampled frames) before or after landing a first real version. `Face.
    source_video_file`/`video_timestamp_seconds` fields + migration, the black-frame check, and
    the actual scheduled task are all still not built -- this remains investigation only, no
    schema or pipeline code written yet.
- **Phase 4 (not started)**: `VideoFileSerializer`/viewset, a `media_type` discriminator for the
  slideshow/frontend to branch `<video>` vs `<img>` (frontend side out of scope for this repo).
- **Phase 5, newly identified (2026-09-07), not started -- transcode pipeline, likely required
  before Phase 4 is actually usable.** Prompted by the user asking about progressive-playback
  slideshow integration. HTML5 `<video>` already supports progressive playback for free via HTTP
  byte-range requests (the browser buffers the first chunk and starts playing before the rest
  streams in) -- but that only works if (a) the codec is browser-playable at all, which most of
  this library's real source codecs are NOT (`.wmv`, `.avi`, `.mts`, `.mpg` all need transcoding
  to H.264/mp4 first -- `.mp4`/`.mov`/`.m4v` from phones may already qualify, not yet checked),
  and (b) the mp4's `moov` atom (metadata) is at the front of the file (`ffmpeg -movflags
  +faststart`), or the browser must download the whole file first just to find it. So Phase 4
  likely can't just be a thin serializer/viewset over the original files -- it probably needs a
  real transcode step (new output files, own disk footprint, own processing time/queue) ahead of
  it. **Deliberately waiting on the full backfill (in progress, see above) to finish before
  scoping this** -- want real numbers (total footage/duration, codec breakdown across the actual
  library) to estimate transcode disk space and processing time before designing this phase.

  **Real numbers now in (2026-09-07), backfill and `VideoFile.file_size_bytes` both done.**
  Added `file_size_bytes` (`BigIntegerField`, migration `filepopulator.0008`, populated via
  `os.path.getsize()` in `create_video_file()`) plus a one-time `backfill_video_file_size`
  management command for rows ingested before the field existed -- deployed (migrate-then-restart,
  additive column, safe) and run for real: **6,946/6,946 rows backfilled, 0 files missing from
  disk**. Real library footprint: **647.6 GB total, 95.5 MB average file size**. Codec x size
  breakdown: h264 366.7GB (4,456 files) + vp9 0.3GB (38 files) = **~367GB already browser-native,
  no transcode needed**; the remaining **~281GB would need transcoding** for universal playback --
  mpeg4 130.5GB (only 96 files -- old high-bitrate camcorder footage, huge per-file average),
  hevc 68.7GB (821 files, Safari-native but not universal), mjpeg 34.4GB (849 files), mpeg2video
  38.0GB (89 files), mpeg1video 8.2GB (411 files), h263/wmv1-3 ~0.9GB combined (186 files). So
  **roughly 57% of the library is already web-playable as-is** -- Phase 5's transcode pipeline
  mainly needs to cover the other ~281GB, not the whole library. Still not designed/started --
  this closes out the scoping-data gap, not the phase itself.

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

`dockerize/requirements.txt` was trimmed and pinned on `backend_upgrade` (both branches have carried the same pinned file for a while now, as of later merges this session -- this note originally said `master` was still the untouched, unbounded `>=` original, which stopped being true a while ago): originally 15 packages removed as "zero references anywhere in the codebase" (`coloredlogs`, `dj-database-url`, `django-celery-beat`, `django-celerybeat-status`, `django-rest-framework` [a dead/unrelated stub package — not `djangorestframework`, which stays], `django-timezone-field`, `ExifRead`, `importlib-metadata`, `pgi`, `piexif`, `psycopg2-pool`, `python-dotenv`, `python-xmp-toolkit`, `SCons`, `twilio`), and every remaining package pinned `==` to the exact version that passed all 93 fast tests. **Correction (2026-08-26)**: `ExifRead`/`piexif` were wrongly included in that "zero references" list and had to be added back — that check only looked for direct imports in our own code, missing that `gpsphoto` (which we do use, for GPS EXIF extraction) imports both itself (`import exifread`, `from piexif import load, dump`) without declaring them in its own `setup.py`'s `install_requires` (empty/broken packaging on GPSPhoto's part). Invisible locally because `picasa_img:latest` already had both installed from before the trim — only surfaced once CI actually ran `pip install -r requirements.txt` into a genuinely clean environment, which is exactly the kind of gap a real CI run is for. Actually-zero-reference removals (the other 13) still stand. Deliberately pinned Django to `6.0.8`, not the newer `6.1` that `pip install --upgrade` offered — upgrading to 6.1 (with scipy bumped to 1.18.1 alongside it) made the test suite hang indefinitely partway through `filepopulator`'s duplicate-detection tests; root cause not confirmed (Django vs. scipy), not chased further, just avoided. If picking this back up: reproduce in a throwaway container (not `picasa_api_dev_test`), and getting a real stack trace will need `--cap-add=SYS_PTRACE` on the container so `py-spy dump` can attach (it couldn't, last time).

**RE-INVESTIGATED 2026-09-05: could not reproduce the hang at all.** Followed the throwaway-container instructions above properly this time (a genuinely separate container off `picasa_img:latest`, `--cap-add=SYS_PTRACE`, not `picasa_api_dev_test`). Two separate tests, both clean:
1. **Runtime upgrade** (`pip install --upgrade django scipy` inside the throwaway container, landing 6.1.1/1.18.1): the specific duplicate-detection tests named in the original report passed individually, the *entire* `filepopulator` fast module passed (83/83, ~103s), and the *entire* fast suite passed (318/318, no hang, exit 0).
2. **A real image build** with `Django==6.1.1`/`scipy==1.18.1` pinned in a scratch copy of `requirements.txt`, using the actual `Dockerfile_picasa` (fresh Ubuntu base, nothing cached from the working image): `pip install -r requirements.txt` — the step that actually installs Django/scipy — completed with zero errors, both times it was run. The build did fail, but at a *later*, unrelated step (`apt install postgresql`, hitting a genuine `404` fetching `libfdisk1` from `security.ubuntu.com` — a stale/out-of-sync Ubuntu package mirror, reproduced identically twice, nothing to do with Django or Python dependencies at all).

**Conclusion: Django 6.1 does not appear to actually be blocked by anything found this session.** Two live explanations for the original report, neither chased further: (a) it was a misdiagnosis of an unrelated environment hiccup (plausibly this same kind of transient apt/mirror issue) at the time, or (b) the actual trigger was specific to the *old* versions of the duplicate-detection tests named in the report -- which were substantially rewritten this session as part of the `DuplicateFile.original` work (see above), so a Django-6.1-specific interaction with that old test code, if real, may simply no longer exist.

**DONE (2026-09-05): upgraded to Django 6.1.1 / scipy 1.18.1 for real, deployed to production.**
`dockerize/requirements.txt` cut over on both branches. Hit the same `libfdisk1` mirror 404 again
on a real deploy-targeted build (not just the earlier scratch test) -- fixed properly this time,
since it would have blocked *any* rebuild of this image going forward, Django or not:
`Dockerfile_picasa` now runs a fresh `apt update` immediately before the `postgresql` install
instead of relying on the one cached at the top of the file from whenever that layer was last
built (Docker's layer caching can leave that index stale relative to what the mirror currently
serves, which is exactly what happened -- a pinned package version had since been pruned from
`security.ubuntu.com`). Validated thoroughly before touching production: built a real image from
`backend_upgrade` off the actual `Dockerfile_picasa`, spun up a dedicated container from it
(separate from `picasa_api_dev_test`), and ran the genuinely complete test suite -- fast and slow,
including real ML inference -- **326/326 passing**. Deployed by rebuilding the real `picasa_img`
via `docker compose build` from `master`'s `dockerize/` and recreating `picasa_api`. Verified live
in production afterward, not just via `manage.py check`: `django.VERSION` confirms `(6, 1, 1, ...)`,
zero unapplied migrations (`migrate --check` clean, no framework-bump migrations needed), and a
real smoke test against `/api/images/` and `/api/person_list/` both returned the expected `403`
for an anonymous request with no errors in the logs. Also corrected a stale doc claim in this same
section: `dockerize/requirements.txt`
on `master` was said to still be the original unbounded `>=` file, untouched deliberately -- it
isn't anymore, both branches have carried the same pinned file for a while now (merged along with
everything else moving between the branches this session).

**RE-INVESTIGATED 2026-09-05: could not reproduce the hang at all.** Followed the throwaway-container instructions above properly this time (a genuinely separate container off `picasa_img:latest`, `--cap-add=SYS_PTRACE`, not `picasa_api_dev_test`). Two separate tests, both clean:
1. **Runtime upgrade** (`pip install --upgrade django scipy` inside the throwaway container, landing 6.1.1/1.18.1): the specific duplicate-detection tests named in the original report passed individually, the *entire* `filepopulator` fast module passed (83/83, ~103s), and the *entire* fast suite passed (318/318, no hang, exit 0).
2. **A real image build** with `Django==6.1.1`/`scipy==1.18.1` pinned in a scratch copy of `requirements.txt`, using the actual `Dockerfile_picasa` (fresh Ubuntu base, nothing cached from the working image): `pip install -r requirements.txt` — the step that actually installs Django/scipy — completed with zero errors, both times it was run. The build did fail, but at a *later*, unrelated step (`apt install postgresql`, hitting a genuine `404` fetching `libfdisk1` from `security.ubuntu.com` — a stale/out-of-sync Ubuntu package mirror, reproduced identically twice, nothing to do with Django or Python dependencies at all).

**Conclusion: Django 6.1 does not appear to actually be blocked by anything found this session.** Two live explanations for the original report, neither chased further: (a) it was a misdiagnosis of an unrelated environment hiccup (plausibly this same kind of transient apt/mirror issue) at the time, or (b) the actual trigger was specific to the *old* versions of the duplicate-detection tests named in the report -- which were substantially rewritten this session as part of the `DuplicateFile.original` work (see above), so a Django-6.1-specific interaction with that old test code, if real, may simply no longer exist. Not yet decided whether to actually cut over `requirements.txt` to 6.1 -- this only re-opens the option, it doesn't commit to it.

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
- [x] **DONE (2026-09-07): `find_and_encode_faces()`'s IOU-matching branch rewritten to use
  optimal bipartite matching instead of hand-rolled case analysis.** Original bug (found
  2026-08-26): when an `ImageFile` already has existing `Face` rows and this run's detector finds
  a different count, the mismatch was handled by several hand-written cases left as hard failures
  by the original author (`NotImplementedError`, `ValueError`, bare `assert`s) for combinations
  believed unreachable but weren't. Per-image containment was fixed same-day (commit `4613c84`) so
  one bad image no longer aborted the whole scheduled batch -- but the underlying logic itself
  stayed broken until now.

  **Investigated before touching anything, per the user's own instinct that this reconciliation
  logic might just be a leftover from the one-time dlib->insightface migration.** That instinct
  was half right: `reset_all_images()`/`starting_reset()` -- the only methods that reset
  `isProcessed=False` in bulk *without* deleting existing faces first -- are referenced (commented
  out) only in `encode_library_to_insightface.py`, exactly the historical migration command, and
  nothing live calls them. But a narrower, still-real live user was found:
  `cleanup_chronically_unmatched_faces` deletes only the *specifically bad* faces on an image,
  leaving other good ones in place, then marks the image for redetection -- a real, if occasional,
  trigger for `n_existing > 0 && n_detect > 0`. Checked the cost of instead just deleting
  everything on affected images (the simpler alternative): among orientation-6/8 images with more
  than one face, 76,325 faces total, 32,771 already validated -- real human work that a
  delete-everything simplification would put at risk. Kept the reconciliation capability rather
  than removing it.

  **Fix**: replaced the whole hand-rolled branch (~170 lines: one-to-one case, "not one-to-one"
  `NotImplementedError`, the `min(max_ious) < thresh` case with its own nested tiebreak-or-raise
  logic, `tiebreak_overlapping_bboxes()` entirely) with `scipy.optimize.linear_sum_assignment`
  (the Hungarian algorithm) on a cost matrix built from the existing/detected box IOUs -- finds the
  single globally-optimal pairing across the whole matrix at once, instead of resolving each
  contested match in isolation via nearest-center-distance. Also fixes a real correctness bug the
  old tiebreak had: when two detected boxes both overlapped one existing face, the losing
  candidate was silently dropped rather than added as its own new face (its column-max was already
  nonzero from overlapping the contested row, so it never qualified as "new" under the old logic
  either) -- confirmed as the likely cause of a real production failure
  (`20220611_102953.jpg`, `"1 is not >= 2"`, 2026-08-26 log). The new code guarantees every
  detected face becomes either a real match update or a new `Face` row -- never silently dropped --
  by construction (any detected column not claimed by an above-threshold assignment is
  unconditionally added as new). Tested: full fast suite 320/320, full `face_manager` slow (real
  ML inference) suite 8/8, including `test_rematched_existing_face_gains_kps_from_update_path`
  (real re-detection against a real image, must rematch existing Face rows by id, not duplicate
  them). Also re-verified directly against the real production case that originally surfaced the
  bug: reset `20220611_102953.jpg` to `isProcessed=False` (2 existing faces, `_NO_FACE_ASSIGNED_`)
  and reprocessed it live with the new code -- see the deploy note below for the result.
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
      for actually using this (grouped review UI) was out of scope for this repo -- confirmed by
      the user (2026-09-05) that it's since been built on the frontend side.
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
- **DONE (2026-09-07): HEIC files with a non-1 EXIF orientation are now handled instead of
  failing loudly.** Found 2026-08-26: a user rotated a real HEIC (`IMG_9370.HEIC`) via an external
  tool that only flipped the orientation tag (to 8) without re-encoding pixels (raw dimensions
  unchanged) -- unlike every real HEIC sample tested when this guard was originally built, where
  libheif always baked the rotation into the pixels at decode and reset the tag to 1.
  `_init_image()`'s HEIC-specific guard (`filepopulator/models.py`) used to deliberately raise
  `OSError` on any non-1 orientation rather than guessing, flagging the file
  `image_load_failed=True` instead of processing it. Before changing anything, visually confirmed
  applying the tag-8 rotation to `IMG_9370.HEIC`'s actual pixels via `apply_exif_orientation()`
  (the same shared helper JPEG already uses) produces the correct upright image -- sent both the
  raw (sideways) and rotated decodes to the user for a direct visual check, confirmed correct.
  Fixed by simply removing the guard's `raise OSError` -- `_init_image()` already calls
  `apply_exif_orientation()` unconditionally further down for every image regardless of format, so
  a non-1 HEIC orientation now just gets rotated like any other image, no HEIC-specific rotation
  logic needed. The separate multi-frame (Live Photo/burst) guard is untouched -- still genuinely
  unsupported, still fails loudly. **Broader validation beyond the one production file**, per the
  user's request: wrote a round-trip test (`test_all_orientation_codes_round_trip_correctly_on_
  real_photos`) that takes each of the 8 real HEIC fixtures (all naturally orientation 1),
  simulates what each of the 7 non-identity EXIF orientation codes' "as-captured" raw pixels would
  look like (the exact mathematical inverse of `apply_exif_orientation()`'s own transform for that
  code), and confirms `apply_exif_orientation()` restores the original upright pixels exactly --
  56 combinations (8 photos x 7 codes), all pixel-exact matches, not just dimension/shape checks.
  Replaced the old `test_orientation_guard_fails_loudly_on_non_one_orientation` test with
  `test_non_one_orientation_is_rotated_and_ingests_successfully` (confirms a non-1-orientation HEIC
  now ingests successfully with correctly swapped width/height, no `FailedImageFile` record).
  Full fast suite: 322/322 passing. Not a JPEG problem: JPEG's orientation-changed case was already
  correctly handled by `create_image_file()`'s "same pixel hash, different orientation" branch
  (stale faces cleared, row updated in place, redetection triggered).
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
- **DONE (2026-09-06): the 29 known duplicate `ImageFile` rows cleaned up in production.** All 29
  were in one folder (`/photos/Pictures_In_Progress/2025/Debbie visit/`), each pair with identical
  `pixel_hash`, `dateModified` (to the microsecond), orientation, and face counts — confirming the
  documented root cause (one batch reprocessing run through that folder hit the now-fixed INSERT-
  instead-of-UPDATE bug for every file). Before deleting, checked every pair for human-entered
  face data unique to the higher-id row (`declared_name` excluding the ignore sentinels) — found
  real validated names on most drop-side rows, but in every single case the identical set of names
  was *also* already present on the keep-side row (a person had independently tagged both visible
  copies the same way), so nothing was at risk of being lost. Kept the lower id in each pair,
  deleted the higher-id row via `ImageFile.delete()` per-instance (not a bulk queryset delete) so
  `Face` cleanup and thumbnail-file removal ran correctly. Verified 0 duplicate filenames remain.
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
- **DONE (2026-09-07): stopped relying on manually-synced cached face-count columns on
  `Person` -- replaced with live queries.** `Person.num_faces`/`num_possibilities`/
  `num_unverified_faces` used to be plain `IntegerField`s kept in sync by
  `increment_assigned()`/`decrement_assigned()`/etc, or recomputed wholesale by the scheduled
  `face_manager.set_face_counts` task -- never a live query, so any code path that mutated
  `Face.declared_name`/`poss_identN` without going through those model methods (e.g. a bulk
  `.update()`) silently left the cached numbers wrong. This bit us directly after the
  `.another_ignore` → `.ignore` merge (2026-08-25) -- `.ignore`'s cached counters went stale until
  manually re-synced. Also folded in a second, related bug: `PersonSerializer.get_num_possibilities`
  was commented out entirely while still declared as a `SerializerMethodField`, so `GET
  /api/people/` (`PersonViewSet`) crashed with `AttributeError` on every request that actually
  serialized a `Person` -- never caught because nothing exercised that endpoint directly until this
  work added a regression test for it.

  **Investigated and benchmarked empirically before implementing** (see the session's own
  benchmarking, not repeated in full here): a naive per-person query loop across all 915 people
  took ~4.5s; a naive multi-`Count(distinct=True)` joined annotation (the "obvious" single-query
  fix) instead produced a Cartesian-product join per person and *hung for 10+ minutes* on real
  data -- confirmed via `pg_stat_activity`, not assumed. Switching to correlated subqueries
  (`Subquery`/`OuterRef`, one per counted relation, no join) avoided that entirely and got the
  single-query version to ~1s. Two new covering indexes,
  `face_manager_face_declared_name_id_covering` (`declared_name_id`) `INCLUDE (id, validated)` and
  `face_manager_face_poss_ident1_id_covering` (`poss_ident1_id`) `INCLUDE (id)`, let Postgres
  answer the counts via index-only scans (`Heap Fetches: 0`, confirmed via `EXPLAIN ANALYZE`)
  instead of bitmap-heap-scanning the wide `face_manager_face` rows (512-d embeddings, keypoints,
  etc.) just to count 3 small columns -- cut it to ~0.2-0.4s single-threaded. Splitting the person
  set across a few threads, each its own DB connection (genuine multi-core parallelism, since each
  Postgres connection is its own backend process -- not fighting Python's GIL), got the full
  915-person roster to ~0.2s at 4 threads. Net: **~4.5s naive loop → ~0.2s live, indexed, and
  parallel** -- fast enough to serve directly from the API with no caching needed.

  **Implementation**: new `face_manager/live_counts.py` -- `annotate_live_face_counts()` (single
  correlated-subquery-annotated queryset, used by `PersonViewSet.get_queryset()`) and
  `compute_live_face_counts()` (the 4-thread chunked version, used by `PersonListView`, which
  computes counts for its full, unpaginated person list up front rather than reading cached
  fields). The two covering indexes ship as a real migration
  (`face_manager/migrations/0010_face_count_covering_indexes.py`, `atomic = False` +
  `CREATE/DROP INDEX CONCURRENTLY IF NOT EXISTS`, since they were already created directly against
  production via `CREATE INDEX CONCURRENTLY` during the investigation -- the migration is a
  documented, idempotent no-op there and a real create for any fresh/CI/dev database).
  `Person.num_faces`/`num_possibilities`/`num_unverified_faces` model fields removed
  (`0009_remove_person_face_counts.py`), along with the six `increment_*`/`decrement_*` bookkeeping
  methods on `Person` and every call site (`Face.associate_person()`/`verify_person_in_image()`/
  `reset_to_pool()`/`remove_poss_ident()`/`set_possible_person()`/`reject_association()` all
  simplified, logic otherwise unchanged), the now-redundant `face_manager.set_face_counts` Celery
  task and its `reset_face_counts` management command, `assign_faces.py`'s post-`execute()`
  "trueing up" recompute pass (and the `ExecuteTrueingUpTests` regression test that existed only to
  cover a since-reverted threading bug in that pass), and the recompute blocks in
  `dedupe_overlapping_faces`/`merge_duplicate_imagefiles` (both now report "computed live, no
  recompute needed" instead). Full fast suite (this stage): 320/320 passing (net change from the
  prior 322 baseline matches exactly: -2 removed counter-bookkeeping tests +1 replacement, -1
  trueing-up test, -1 removed dedupe-recompute test, +1 new `/api/people/` regression test).

  **Deployed same day, and a real second bug found during production smoke-testing.** Sequenced
  as: restart `picasa_api` first (new code, still-old DB schema -- safe, since the new code no
  longer references the 3 columns at all), *then* `manage.py migrate` (drops the columns +
  no-ops the already-existing covering indexes) -- deliberately the opposite order of the naive
  "migrate then restart," which would have left the *old*, still-running code erroring on every
  query the instant the columns vanished. Verified via `manage.py migrate --check` (clean) and a
  direct query confirming the columns were gone.

  Smoke-testing the two real endpoints after deploy surfaced a second, more serious bug that
  fixing the `get_num_possibilities` crash had been *masking*: `PersonSerializer.face_declared`
  (a `FaceSubsetSerializer(many=True)`) nested-serializes every one of a person's `Face` rows as
  its own hyperlinked URL. For a huge-gallery person -- the blank sentinel (`_NO_FACE_ASSIGNED_`,
  ~99k faces) or `.ignore` (~131k faces), both of which land on `/api/people/`'s default
  (unordered) first page -- that means building six-figure numbers of URLs in pure Python on a
  single request. Confirmed via a real request: no DB query even running (checked
  `pg_stat_activity` -- empty), 85+ seconds elapsed, 7GB+ RSS, had to be `kill -9`'d. This was
  always latent in `PersonSerializer` but never actually reachable before this session's fix,
  since every prior request 500'd on the `get_num_possibilities` crash before ever reaching
  `face_declared` -- fixing the crash without also addressing this would have traded "instant
  500" for "worker hangs 85+ seconds and OOMs," a regression in production stability, not an
  improvement. **Fixed by removing the dangerous surface instead of trying to paginate/limit
  it**, per the user's explicit call once this was explained: `PersonViewSet` is now a bare
  `GenericViewSet` (not `ModelViewSet`) -- no `list`/`create`/`retrieve`/`update`/`destroy` at
  all, so there's nothing left that can trigger `face_declared`'s nested serialization. Its two
  real, actually-used actions (`rename`, `toggle_further_unlikely`, both hand-rolled plain-dict
  responses that never touched `PersonSerializer` or `face_declared` in the first place) are
  unaffected -- `GenericViewSet` still provides `get_object()`/`get_queryset()`, which both rely
  on. `PersonSerializer` itself was deleted entirely (nothing else referenced it once the
  viewset stopped using it) -- `FaceSubsetSerializer` (its nested field's type) is kept, since a
  *different*, safe usage of it still exists (`ImageFileSerializer.face_set`, bounded by "faces
  per photo," a completely different scale than "faces per person"). Regression test
  (`test_list_and_retrieve_are_not_exposed`) confirms `GET /api/people/` and `GET
  /api/people/<id>/` now 404 (a router only wires up a URL for an action that actually exists --
  not DRF's 405, since the route itself doesn't exist at all). Full fast suite after this second
  fix: 320/320 passing (net zero test-count change: one test rewritten, not added/removed).
  Deployed the same way as the first stage (code change only this time, no new migration --
  `picasa_api` restart alone was sufficient).
- **DONE (2026-09-07): `set_possible_person()`/`reject_association()` no longer hardcode `5`/
  `range(1, 6)` via `eval`/`exec`.** Both now use `self.NUM_POSSIBLE_IDENTITIES` for every bound
  (the `poss_idx` range assert, the candidate-scan loop, the compaction loop, the final-clear
  loop) and plain `setattr()`/`getattr()` instead of building and `eval`/`exec`-ing an f-string --
  same pattern `remove_poss_ident()` already used since its own earlier fix. Logic unchanged, pure
  mechanical swap; `Face.NUM_POSSIBLE_IDENTITIES` is already the single source of truth checked
  against the model's actual `poss_identN`/`weight_N` field pairs by a Django system check
  (`face_manager.E001`), so this closes the last gap where changing that constant wouldn't have
  been enough on its own. Full fast suite: 320/320 passing (no test changes needed -- existing
  coverage already exercises both methods' behavior, which is unchanged).
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
- **FIXED (2026-09-05): `ImageFile.save()`'s unconditional MD5 rehash.** Traced through the real
  cost breakdown before touching anything (measured against a real fixture): `_init_image()`
  (decode + EXIF orientation) is cheap, ~0.5ms; `_generate_md5_hash()` (pixel hash + phash + a DB
  query) is the actual expensive part, ~15ms steady-state per call. Checked every real call site
  in the codebase (`create_image_file()`'s several branches -- new file, unchanged-pixel-hash
  update, orientation change, moved file -- plus `add_file_manual.py`, which routes through the
  same function) and confirmed every single one already computes and verifies a correct
  `pixel_hash` *before* calling `.save()` -- so `.save()`'s own unconditional rehash was pure
  repeated work in the common case of "file's mtime changed but content didn't," redoing the
  exact same decode+hash a second time for a value already known correct. Fixed: `save()` now
  only calls `_generate_md5_hash()` when `pixel_hash` isn't already a real value (still `-1`, the
  field's default); otherwise it just calls the new, cheap `_refresh_file_hash()` (a filename-only
  hash, no image decode) so `file_hash` -- used to build the thumbnail path -- stays correct even
  when a file gets moved to a new path without its pixel content changing. 3 new tests (skips
  rehash when already set, still computes when not, `file_hash` refreshes correctly on the
  skip-rehash path after a simulated move). Full fast suite: 321/321 passing. `backfill_phash`
  (below) already worked around this same cost by deliberately not going through `.save()` at
  all -- that workaround is now less necessary but still harmless to leave as-is.
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
