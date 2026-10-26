from django.contrib import admin
import admin_thumbnails

# Register your models here.

from .models import ImageFile, Directory, PhotoCollection

# Set the thumbnail style for the image files. thumbnail_big/medium were
# removed 2026-10-08 (zero real consumers -- see CLAUDE.md); the admin
# preview now uses thumbnail_small (100x100) instead -- smaller than the
# old 500px preview, but this isn't part of the real review workflow
# (that's all frontend), just an admin-page convenience.
admin_thumbnails.ADMIN_THUMBNAIL_STYLE = {'display': 'block', 'width': '100px', 'height': 'auto'}

# Define how the list of all of the file objects will look.
def obj_filename(obj):
    return obj.filename
obj_filename.short_description = 'Filename'

def directory_names(obj):
    return obj.dir_path
directory_names.short_description = "Folder"

# Use a decorator to say that the thumbnail should be thumbnail_small
# (the only thumbnail size still generated -- see above).
@admin_thumbnails.thumbnail('thumbnail_small')
class ImageFileAdmin(admin.ModelAdmin):

    # Set the value for fields that are empty.
    # Part of extending ModelAdmin
    empty_value_display = '-empty-'

    list_display = (obj_filename,)
    # Set fields that are readonly in admin
    readonly_fields = ('pixel_hash', 'file_hash', 'iso_value', 'filename', 'directory', 'dateAdded', \
    				'dateModified', 'dateTaken', 'dateTakenValid', 'width', 'height', 'orientation', \
    				'camera_make', 'camera_model', 'flash_info', 'iso_value', 'light_source')

    # Set basic and advanced options.
    fieldsets = (
        (None, {
            'fields': ('filename', 'directory', 'thumbnail_small_thumbnail', 'dateAdded', \
            	'dateModified', 'dateTaken', 'dateTakenValid', 'width', 'height', 'orientation')
        }),

        ('Advanced options', {
            'classes': ('collapse',),
            'fields': ('pixel_hash', 'file_hash',  'camera_make', 'camera_model', \
                'flash_info',  'iso_value', 'light_source', 'gps_lat_decimal', \
                'gps_lon_decimal', 'isProcessed'),
        }),
    )

class DirectoryAdmin(admin.ModelAdmin):
    empty_value_display = '-empty-'
    # Set how directories will be displayed
    list_display = (directory_names,)

    # Make all fields read-only
    readonly_fields = ('dir_path', 'mean_datetime', 'first_datetime', 'mean_datesec', 'first_datesec')

    # May be nice one day to include a link to all 
    # images in this directory, but I don't want to do that right now.

class PhotoCollectionAdmin(admin.ModelAdmin):
    empty_value_display = '-empty-'
    list_display = ('name', 'root_directory', 'is_default', 'min_face_det_score', 'min_face_relative_size')
    # root_directory/is_default are structural -- changing them doesn't
    # re-scope already-ingested images (Directory.collection is only
    # resolved once, at Directory creation time, via resolve_for_path())
    # and could silently orphan a collection's existing content, so
    # they're read-only here rather than editable by accident. The two
    # face-detection floors are the actual point of this admin page --
    # PhotoCollection.save() already dispatches apply_collection_floor_
    # change() on a real change, same as any other save() path, so
    # editing them here needs no special handling.
    readonly_fields = ('name', 'root_directory', 'is_default', 'created_at', 'updated_at')

# Make these class extensions the default for displaying these
# types in the admin panel.
admin.site.register(ImageFile, ImageFileAdmin)
admin.site.register(Directory, DirectoryAdmin)
admin.site.register(PhotoCollection, PhotoCollectionAdmin)


