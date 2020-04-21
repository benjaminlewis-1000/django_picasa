from django.contrib import admin
import admin_thumbnails

# Register your models here.

from .models import ImageFile, Directory

# Set the thumbnail style for the image files, with
# a width of 500px. We define the thumbnail_big
# field as the one that is shown in the admin panel.
# We omit the medium and small thumbnails since
# that would be redundant.
admin_thumbnails.ADMIN_THUMBNAIL_STYLE = {'display': 'block', 'width': '500px', 'height': 'auto'}

# Define how the list of all of the file objects will look.
def obj_filename(obj):
    return obj.filename
obj_filename.short_description = 'Filename'

def directory_names(obj):
    return obj.dir_path
directory_names.short_description = "Folder"

# Use a decorator to say that the thumbnail should be
# thumbnail-big. Other thumbnails are redundant and smaller.
@admin_thumbnails.thumbnail('thumbnail_big')
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
            'fields': ('filename', 'directory', 'thumbnail_big_thumbnail', 'dateAdded', \
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

# Make these class extensions the default for displaying these
# types in the admin panel.
admin.site.register(ImageFile, ImageFileAdmin)
admin.site.register(Directory, DirectoryAdmin)


