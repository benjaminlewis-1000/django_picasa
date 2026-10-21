#! /usr/bin/env python

from django.conf import settings
from django.contrib.auth.models import User, Group
from django.core.paginator import Paginator
from drf_queryfields import QueryFieldsMixin
from api.models import ScheduleRule, SlideshowConfig
from face_manager.models import Person, Face
from filepopulator.models import ImageFile, Directory, VideoFile, PhotoCollection
from rest_framework import serializers
import datetime
import dateutil.parser
import json
import os
from django.db.models import Count, Q, Sum


def _format_duration_seconds(total_seconds):
    """'Dd HH:MM:SS' -- days only included when nonzero, since most of
    this library's individual videos/aggregates are well under a day
    but the unprocessed-backlog total is not."""
    total_seconds = int(round(total_seconds or 0))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f'{days}d {hours:02d}:{minutes:02d}:{seconds:02d}'
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class FaceSubsetSerializer(QueryFieldsMixin, serializers.HyperlinkedModelSerializer):
    
    # face_name = serializers.SerializerMethodField()

    class Meta:

        model = Face
        # fields = ['url', 'source_image_file', 'face_thumbnail', \
        #     'declared_name', 'face_name']
        fields = ['url']

    # def get_face_name(self, obj):
    #     # face_declared is the related field of the ForeignKey.
    #     return obj.declared_name.person_name


class UserSerializer(serializers.HyperlinkedModelSerializer):
	"""docstring for UserSerializer"""
	class Meta:
		model = User
		fields = ['url', 'username', 'email', 'groups']


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ['url', 'name']

class ImageFileSerializer(serializers.HyperlinkedModelSerializer):

    parsed_date = serializers.SerializerMethodField()
    face_set = FaceSubsetSerializer(read_only = True, many=True)

    def get_parsed_date(self, obj):
        first_date = obj.dateTaken
        
        data = {'year': first_date.year, 
                'month': first_date.month,
                'day': first_date.day, 
                'hour': first_date.hour,
                'minute': first_date.minute, 
                'sec': first_date.second}

        return data

    class Meta:

        model = ImageFile
        fields = ['url', 'id', 'filename', 'pixel_hash', 'file_hash', 'camera_make', 'camera_model', \
                  'flash_info', 'exposure_num', 'exposure_denom', 'focal_num', 'focal_denom', \
                  'fnumber_num', 'fnumber_denom', 'iso_value', 'light_source', 'gps_lat_decimal', \
                  'gps_lon_decimal', 'thumbnail_small', \
                  'full_res_path',  'exposure', 'directory', 'dateTaken', 'tags', 'description', 'isProcessed', \
                  'parsed_date', 'face_set']

class ImageSummarySerializer(serializers.HyperlinkedModelSerializer):

    class Meta:

        model = ImageFile
        fields = ['url', 'filename', 'thumbnail_small', \
                  'directory']


class VideoFileSerializer(serializers.HyperlinkedModelSerializer):
    """Phase 4 (slideshow serving). Mirrors ImageFileSerializer's shape
    where the concept overlaps (parsed_date, directory, thumbnail), plus
    video-specific fields the frontend needs to decide what/whether to
    play. playback_url is computed, not stored -- see get_playback_url's
    own comment for why it can't just be VideoFile.playback_path
    directly."""

    parsed_date = serializers.SerializerMethodField()
    playback_url = serializers.SerializerMethodField()
    ready = serializers.SerializerMethodField()

    def get_parsed_date(self, obj):
        first_date = obj.dateTaken
        return {'year': first_date.year,
                'month': first_date.month,
                'day': first_date.day,
                'hour': first_date.hour,
                'minute': first_date.minute,
                'sec': first_date.second}

    def get_ready(self, obj):
        return obj.get_playback_url() is not None

    def get_playback_url(self, obj):
        return obj.get_playback_url()

    class Meta:

        model = VideoFile
        fields = ['url', 'id', 'filename', 'directory', 'dateTaken', 'parsed_date', \
                  'duration_seconds', 'width', 'height', 'rotation', 'codec', \
                  'thumbnail_small', 'playback_url', 'ready', 'isProcessed']


class DirectorySerializer(QueryFieldsMixin, serializers.HyperlinkedModelSerializer):
    
    # The ForeignKey in the ImageFile that points to directory needs to have
    # the related_name parameter set to 'image_set'. 
    
    # The 'image_set' is set up with the related_name='image_set' argument in the 
    # Django model's ForeignKey.
    image_set = ImageSummarySerializer(read_only=True, many=True)
    num_images = serializers.SerializerMethodField()
    year = serializers.SerializerMethodField()
    month = serializers.SerializerMethodField()

    def get_num_images(self, obj):
        # image_set is the related field of the ForeignKey.
        return obj.image_set.count()

    def get_year(self, obj):
        first_date = obj.first_datesec
        return datetime.datetime.fromtimestamp(first_date).year

    def get_month(self, obj):
        first_date = obj.first_datesec
        return datetime.datetime.fromtimestamp(first_date).month

    class Meta:

    # Able to filter to get just the fields you want usin QueryFieldsMixin.
    # https://github.com/wimglenn/djangorestframework-queryfields
    # Suggested for list of all directories: 
    # <URL>/api/directories/?fields!=imgs_in_dir
        model = Directory
        fields = ['url', 'id', 'dir_path', 'top_level_name', 'mean_datetime', 'mean_datesec', \
        'first_datetime', 'first_datesec', 'num_images', 'year', 'month', 'image_set']
#        fields = ['dir_path', 'top_level_name', 'mean_datetime', 'mean_datesec', \
#        'first_datetime', 'first_datesec', ]

class FaceSerializer(QueryFieldsMixin, serializers.HyperlinkedModelSerializer):

    face_name = serializers.SerializerMethodField(read_only=True)
    poss_face_name1 = serializers.SerializerMethodField(read_only=True)
    poss_face_name2 = serializers.SerializerMethodField(read_only=True)
    poss_face_name3 = serializers.SerializerMethodField(read_only=True)
    poss_face_name4 = serializers.SerializerMethodField(read_only=True)
    poss_face_name5 = serializers.SerializerMethodField(read_only=True)

    def get_face_name(self, obj):
        # num_calls += 1
        # face_declared is the related field of the ForeignKey.
        return obj.declared_name.person_name

    def get_poss_face_name1(self, obj):
        if obj.poss_ident1 is not None:
            return obj.poss_ident1.person_name
        else:
            return "N/A"

    def get_poss_face_name2(self, obj):
        if obj.poss_ident2 is not None:
            return obj.poss_ident2.person_name
        else:
            return "N/A"
            
    def get_poss_face_name3(self, obj):
        if obj.poss_ident3 is not None:
            return obj.poss_ident3.person_name
        else:
            return "N/A"

    def get_poss_face_name4(self, obj):
        if obj.poss_ident4 is not None:
            return obj.poss_ident4.person_name
        else:
            return "N/A"

    def get_poss_face_name5(self, obj):
        if obj.poss_ident5 is not None:
            return obj.poss_ident5.person_name
        else:
            return "N/A"


    class Meta:

        model = Face
        # No face_encoding in this api -- it's not necessary. 
        fields = ['url', 'source_image_file', 'written_to_photo_metadata',  \
            'declared_name', 'face_name', \
            'poss_ident1', 'poss_face_name1', 'weight_1', \
            'poss_ident2', 'poss_face_name2', 'weight_2', \
            'poss_ident3', 'poss_face_name3', 'weight_3', \
            'poss_ident4', 'poss_face_name4', 'weight_4', \
            'poss_ident5', 'poss_face_name5', 'weight_5', \
            'box_top', 'box_bottom', 'box_left', \
            'box_right', 'face_thumbnail', 'rejected_fields', 'validated']
        # fields = ['url', 'source_image_file', 'written_to_photo_metadata',  \
        #     'declared_name', 'face_name', 'poss_ident1', 'weight_1', 'poss_face_name1', \
        #     'poss_ident2', 'poss_face_name2', 'poss_ident3',  'poss_ident4', \
        #     'poss_ident5', 'box_top', 'box_bottom', 'box_left', \
        #     'box_right', 'face_thumbnail']




# PersonSerializer removed -- it exposed face_declared as a fully nested
# list of hyperlinked Face URLs, which for a large-gallery person (the
# blank sentinel, ~99k faces; .ignore, ~131k faces) meant building
# six-figure numbers of URLs in pure Python on a single request. This was
# always latent but never reachable in practice: get_num_possibilities
# was commented out while still declared as a SerializerMethodField, so
# every request 500'd before ever reaching face_declared. See
# PersonViewSet (api/views.py) -- now a bare GenericViewSet exposing only
# its two real actions (rename, toggle_further_unlikely), neither of
# which used this serializer.

# Source : https://medium.com/django-rest-framework/django-rest-framework-viewset-when-you-don-t-have-a-model-335a0490ba6f
class ParameterSerializer(serializers.Serializer):
    img_server_user = serializers.CharField(max_length=256)
    img_server_pw = serializers.CharField(max_length=256)
    random_access_key = serializers.CharField(max_length=256)

    class Parameters(object):
        def __init__(self):
            self.img_server_pw = settings.MEDIA_URL_PW   
            self.img_server_user = settings.MEDIA_URL_USER
            self.img_server_address = settings.MEDIA_URL
            self.random_access_key = settings.RANDOM_ACCESS_KEY

    def create(self, validated_data):
        return Parameters()
    
def _compute_video_stats(video_qs):
    """Shared by ServerStatsSerializer's global totals and its
    per-collection breakdown -- same formulas, just handed a pre-filtered
    (or unfiltered, for the global case) VideoFile queryset. Scoped via
    VideoFile.directory__collection (the single "home" FK -- videos have
    no ImageFile.collections-style "visible from" M2M, since there's no
    cross-collection video dedup/cross-linking the way there is for
    images)."""
    stats = {}
    stats['num_videos'] = video_qs.count()
    stats['num_videos_processed'] = video_qs.filter(isProcessed=True).count()
    video_percent = (stats['num_videos_processed'] / stats['num_videos'] * 100) if stats['num_videos'] else 0.0
    stats['percent_video_processed'] = f'{video_percent:.2f}%'

    total_seconds = video_qs.aggregate(total=Sum('duration_seconds'))['total'] or 0
    unprocessed_seconds = video_qs.filter(isProcessed=False).aggregate(
        total=Sum('duration_seconds'))['total'] or 0
    stats['total_video_length'] = _format_duration_seconds(total_seconds)
    stats['unprocessed_video_length'] = _format_duration_seconds(unprocessed_seconds)
    processed_seconds = total_seconds - unprocessed_seconds
    length_percent = (processed_seconds / total_seconds * 100) if total_seconds else 0.0
    stats['percent_video_length_processed'] = f'{length_percent:.2f}%'

    # Distinct from FailedVideoFile (an ingestion-time failure -- never
    # even became a VideoFile row) -- this counts a VideoFile that WAS
    # ingested but whose face-extraction pass raised (see
    # face_manager.tasks.process_video_faces).
    stats['num_videos_failed'] = video_qs.filter(face_extraction_failed=True).count()

    return stats


def _sentinel_person_ids():
    """Resolves BLANK_FACE_NAME/IGNORED_NAMES to Person ids once per
    request instead of letting every _compute_image_face_stats() call
    join Face all the way to the (small, but not free) Person table just
    to match a sentinel *name* -- filtering Face directly on the
    already-resolved declared_name_id avoids that join entirely, on
    every one of the several counts below."""
    blank_id = Person.objects.filter(person_name=settings.BLANK_FACE_NAME).values_list('id', flat=True).first()
    ignored_ids = list(Person.objects.filter(person_name__in=settings.IGNORED_NAMES).values_list('id', flat=True))
    return blank_id, ignored_ids


def _compute_image_face_stats(image_qs, face_qs, blank_person_id, ignored_person_ids):
    """Shared by ServerStatsSerializer's global totals and its
    per-collection breakdown -- same formulas, just handed a
    pre-filtered (or unfiltered, for the global case) ImageFile/Face
    queryset pair.

    Both querysets are backed by real production-scale tables (the
    per-collection Face queryset in particular has to JOIN through
    ImageFile and the collections M2M table against a 700k+-row table) --
    measured at several seconds total when each of the stats below ran as
    its own separate .count() query, since every one of them repeated
    that same expensive join from scratch. Collapsed here into two
    queries total per call: one combined ImageFile aggregate, and one
    combined conditional-count Face aggregate (so the Face-side join only
    ever gets paid for once). num_people is deliberately its OWN query,
    not folded into that combined aggregate -- EXPLAIN ANALYZE showed
    Postgres plans COUNT(DISTINCT ...) inside a multi-aggregate query as
    a disk-spilling sort of the whole result set, while a separate
    .values().distinct().count() plans as a cheap hash-based dedup
    instead; measured faster standalone than combined, for real."""
    stats = {}

    image_agg = image_qs.aggregate(
        num_imgs=Count('id'),
        num_face_processed=Count('id', filter=Q(isProcessed=True)),
    )
    stats['num_imgs'] = image_agg['num_imgs']
    stats['num_face_processed'] = image_agg['num_face_processed']
    proc_percent = (stats['num_face_processed'] / stats['num_imgs'] * 100) if stats['num_imgs'] else 0.0
    stats['percent_face_processed'] = f'{proc_percent:.2f}%'

    stats['num_people'] = face_qs.values('declared_name_id').distinct().count()

    face_agg = face_qs.aggregate(
        num_faces=Count('id'),
        num_unlabeled_faces=Count('id', filter=Q(declared_name_id=blank_person_id)),
        num_verified=Count('id', filter=Q(validated=True)),
        num_tagged=Count('id', filter=~Q(declared_name_id__in=ignored_person_ids)),
    )
    stats['num_faces'] = face_agg['num_faces']
    stats['num_unlabeled_faces'] = face_agg['num_unlabeled_faces']
    stats['num_verified'] = face_agg['num_verified']
    stats['num_tagged'] = face_agg['num_tagged']
    verified_percent = (stats['num_verified'] / stats['num_tagged'] * 100) if stats['num_tagged'] else 0.0
    stats['verified_progress'] = f'{verified_percent:.2f}%'

    return stats


def _compute_by_collection_stats(collections, blank_person_id, ignored_person_ids):
    """Per-collection breakdown, restructured to join Face<->ImageFile<->
    the collections M2M table ONCE for ALL collections (via .values(
    'collections').annotate(...), i.e. GROUP BY collection) instead of
    once per collection as _compute_image_face_stats() does for the
    global case -- measured as a real win (e.g. the two conditional-count
    aggregates below: ~0.8s combined across every collection, vs ~2s+ for
    the same work done as N separate per-collection queries) that gets
    BETTER, not worse, as more collections are added, since it's one join
    regardless of collection count.

    num_people is deliberately excluded from this grouped shape and stays
    a separate, per-collection query -- measured GROUPed COUNT(DISTINCT)
    as dramatically worse (~3s) than N separate ungrouped distinct counts
    (~1.2s for 2 collections combined), for the same reason plain
    COUNT(DISTINCT) is avoided in _compute_image_face_stats' own combined
    aggregate (Postgres plans it as a disk-spilling sort)."""
    image_rows = ImageFile.objects.filter(collections__isnull=False).values('collections').annotate(
        num_imgs=Count('id'),
        num_face_processed=Count('id', filter=Q(isProcessed=True)),
    )
    image_by_collection = {row['collections']: row for row in image_rows}

    face_rows = Face.objects.filter(source_image_file__collections__isnull=False).values(
        'source_image_file__collections'
    ).annotate(
        num_faces=Count('id'),
        num_unlabeled_faces=Count('id', filter=Q(declared_name_id=blank_person_id)),
        num_verified=Count('id', filter=Q(validated=True)),
        num_tagged=Count('id', filter=~Q(declared_name_id__in=ignored_person_ids)),
    )
    face_by_collection = {row['source_image_file__collections']: row for row in face_rows}

    results = {}
    for collection in collections:
        stats = {}

        image_row = image_by_collection.get(
            collection.id, {'num_imgs': 0, 'num_face_processed': 0})
        stats['num_imgs'] = image_row['num_imgs']
        stats['num_face_processed'] = image_row['num_face_processed']
        proc_percent = (stats['num_face_processed'] / stats['num_imgs'] * 100) if stats['num_imgs'] else 0.0
        stats['percent_face_processed'] = f'{proc_percent:.2f}%'

        stats['num_people'] = Face.objects.filter(
            source_image_file__collections=collection).values('declared_name_id').distinct().count()

        face_row = face_by_collection.get(collection.id, {
            'num_faces': 0, 'num_unlabeled_faces': 0, 'num_verified': 0, 'num_tagged': 0})
        stats['num_faces'] = face_row['num_faces']
        stats['num_unlabeled_faces'] = face_row['num_unlabeled_faces']
        stats['num_verified'] = face_row['num_verified']
        stats['num_tagged'] = face_row['num_tagged']
        verified_percent = (stats['num_verified'] / stats['num_tagged'] * 100) if stats['num_tagged'] else 0.0
        stats['verified_progress'] = f'{verified_percent:.2f}%'

        results[collection.id] = stats

    return results


class ServerStatsSerializer(serializers.Serializer):
    num_imgs = serializers.IntegerField()
    num_face_processed = serializers.IntegerField()
    percent_face_processed = serializers.CharField(max_length=256)
    num_people = serializers.IntegerField()
    num_faces = serializers.IntegerField()
    estimated_hours_facerec_left = serializers.FloatField()
    imgs_per_hour_est = serializers.IntegerField()
    num_unlabeled_faces = serializers.IntegerField()
    num_verified = serializers.IntegerField()
    num_tagged = serializers.IntegerField()
    verified_progress = serializers.CharField(max_length=256)
    num_videos = serializers.IntegerField()
    num_videos_processed = serializers.IntegerField()
    percent_video_processed = serializers.CharField(max_length=256)
    total_video_length = serializers.CharField(max_length=256)
    unprocessed_video_length = serializers.CharField(max_length=256)
    percent_video_length_processed = serializers.CharField(max_length=256)
    num_videos_failed = serializers.IntegerField()
    # Same fields as above (minus the video-only ones, which have no
    # PhotoCollection concept to scope by), broken out per collection --
    # a plain list of dicts rather than a nested serializer, since this
    # is a read-only reporting shape with no input to validate.
    by_collection = serializers.ListField(child=serializers.DictField(), read_only=True)

    class Stats(object):
        def __init__(self):
            blank_person_id, ignored_person_ids = _sentinel_person_ids()

            global_stats = _compute_image_face_stats(
                ImageFile.objects.all(), Face.objects.all(), blank_person_id, ignored_person_ids)
            self.num_imgs = global_stats['num_imgs']
            self.num_face_processed = global_stats['num_face_processed']
            self.percent_face_processed = global_stats['percent_face_processed']
            self.num_people = global_stats['num_people']
            self.num_faces = global_stats['num_faces']
            self.num_unlabeled_faces = global_stats['num_unlabeled_faces']
            self.num_verified = global_stats['num_verified']
            self.num_tagged = global_stats['num_tagged']
            self.verified_progress = global_stats['verified_progress']

            self.imgs_per_hour_est = 1600 # An estimate -- I'm clocking 177 images in 26 minutes
            imgs_left_to_process = self.num_imgs - self.num_face_processed
            self.estimated_hours_facerec_left = imgs_left_to_process / self.imgs_per_hour_est

            global_video_stats = _compute_video_stats(VideoFile.objects.all())
            self.num_videos = global_video_stats['num_videos']
            self.num_videos_processed = global_video_stats['num_videos_processed']
            self.percent_video_processed = global_video_stats['percent_video_processed']
            self.total_video_length = global_video_stats['total_video_length']
            self.unprocessed_video_length = global_video_stats['unprocessed_video_length']
            self.percent_video_length_processed = global_video_stats['percent_video_length_processed']
            self.num_videos_failed = global_video_stats['num_videos_failed']

            # Per-collection breakdown. Images: scoped via
            # ImageFile.collections (the "visible from" M2M), not
            # Directory.collection (the single "home" FK), so a cross-
            # collection duplicate photo is correctly counted in every
            # collection it's visible from, not just the one that
            # originally discovered it. Videos: scoped via
            # VideoFile.directory__collection -- there's no M2M-style
            # "visible from" concept on the video side (see
            # _compute_video_stats' own docstring), and resolving that
            # FK at all required a real fix (create_video_file() never
            # used to set Directory.collection -- see
            # backfill_video_directory_collection, 2026-10-21).
            collections = list(PhotoCollection.objects.all().order_by('name'))
            by_collection_stats = _compute_by_collection_stats(
                collections, blank_person_id, ignored_person_ids)

            self.by_collection = []
            for collection in collections:
                collection_stats = by_collection_stats[collection.id]

                video_qs = VideoFile.objects.filter(directory__collection=collection)
                collection_stats.update(_compute_video_stats(video_qs))

                collection_stats['name'] = collection.name
                collection_stats['is_default'] = collection.is_default
                self.by_collection.append(collection_stats)

    def create(self, validated_data):
        return Stats()


class SlideshowConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = SlideshowConfig
        fields = [
            'id', 'name', 'people', 'year_start', 'year_end', 'cronological',
            'slide_len', 'show_meta', 'include_videos', 'include_images',
            'debug_limit', 'is_default', 'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def validate(self, data):
        year_start = data.get('year_start', getattr(self.instance, 'year_start', None))
        year_end = data.get('year_end', getattr(self.instance, 'year_end', None))
        if year_start is not None and year_end is not None and year_start > year_end:
            raise serializers.ValidationError('year_start must not be after year_end.')
        return data


class ScheduleRuleSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScheduleRule
        fields = [
            'id', 'config', 'name', 'days', 'start_time', 'end_time',
            'priority', 'enabled', 'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def validate_days(self, value):
        days = [d.strip().upper() for d in value.split(',') if d.strip()]
        if not days:
            raise serializers.ValidationError('At least one day is required.')
        invalid = sorted(set(days) - set(ScheduleRule.WEEKDAY_ABBREVIATIONS))
        if invalid:
            raise serializers.ValidationError(
                f"Invalid day(s): {', '.join(invalid)}. Use MON/TUE/WED/THU/FRI/SAT/SUN."
            )
        # Normalize to a fixed MON..SUN order regardless of input order/
        # duplicates, so two equivalent day sets always compare equal
        # (e.g. in tests) and render consistently in the settings page.
        ordered = [d for d in ScheduleRule.WEEKDAY_ABBREVIATIONS if d in days]
        return ','.join(ordered)


class CameraGpsBackfillStatusSerializer(serializers.Serializer):
    """One row per distinct ImageFile.camera_model (grouped by model
    ALONE, not (make, model) together -- the same physical device can
    write different camera_make spellings across firmware/OS versions,
    e.g. 'LGE'/'LG Electronics' both for 'LM-G710', confirmed via a real
    10,063-vs-115-photo split that grouping by the pair would have
    silently fragmented). camera_make_examples lists every distinct make
    string seen for this model, for frontend display context only."""
    camera_model = serializers.CharField()
    camera_make_examples = serializers.ListField(child=serializers.CharField())
    total = serializers.IntegerField()
    missing_gps = serializers.IntegerField()
    enabled = serializers.BooleanField()
