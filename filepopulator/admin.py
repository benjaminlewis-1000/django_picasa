from django.contrib import admin
import admin_thumbnails

# Register your models here.

from .models import ImageFile, Directory

# admin.site.register(Face)

list_display= ('image_img','ImageFile',)
readonly_fields = ('image_img',)

admin_thumbnails.ADMIN_THUMBNAIL_STYLE = {'display': 'block', 'width': '500px', 'height': 'auto'}
# admin_thumbnails.ADMIN_THUMBNAIL_FIELD_SUFFIX = ''

def upper_case_name(obj):
    return "Helll ther!"
upper_case_name.short_description = 'Name'

@admin_thumbnails.thumbnail('thumbnail_big')
# @admin_thumbnails.thumbnail('thumbnail_medium')
# @admin_thumbnails.thumbnail('thumbnail_small')
class ImageFileAdmin(admin.ModelAdmin):
    pass
    empty_value_display = '-empty-'
    # list_display = ('thumbnail_big_thumbnail','thumbnail_medium_thumbnail','thumbnail_small_thumbnail',)
    list_display = (upper_case_name,)
    readonly_fields = ('pixel_hash', 'file_hash', 'iso_value', 'filename', 'directory', 'dateAdded', \
    				'dateModified', 'dateTaken', 'dateTakenValid', 'width', 'height', 'orientation', \
    				'camera_make', 'camera_model', 'flash_info', 'iso_value', 'light_source')

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

admin.site.register(ImageFile, ImageFileAdmin)
admin.site.register(Directory)


# exposure_num
# exposure_denom
# focal_num
# focal_denom
# fnumber_num
# fnumber_denom
# # Default
# dateAdded
# dateModified
# width
# height
# # Default
# dateTaken
# dateTakenValid
# # isProcessed
# isProcessed
# orientation