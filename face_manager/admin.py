from django.contrib import admin
import admin_thumbnails

# Register your models here.

from .models import Face, Person

# Set the thumbnail style for the image files, with
# a width of 300px. We define the thumbnail_big
# field as the one that is shown in the admin panel.
# We omit the medium and small thumbnails since
# that would be redundant.
admin_thumbnails.ADMIN_THUMBNAIL_STYLE = {'display': 'block', 'width': '300px', 'height': 'auto'}

def person_obj(obj):
    return obj.person_name

# Define a custom Admin panel for faces. We 
# use the decorator to say that the admin page thumbnail
# will come from face_thumbnail. The second
# parameter is what it is called in the admin panel,
# instead of "Preview".
@admin_thumbnails.thumbnail('face_thumbnail', 'face')
class FaceAdmin(admin.ModelAdmin):
    # Set empty value
    empty_value_display = '-empty-'

    readonly_fields=('source_image_file', 'face_encoding', 'poss_ident1', \
        'poss_ident2', 'poss_ident3', 'poss_ident4', 'poss_ident5', 'box_top', \
        'box_left', 'box_right', 'box_bottom', 'written_to_photo_metadata', \
        'face_thumbnail')

    # Set basic and advanced options.
    fieldsets = (
        (None, {
            'fields': ('declared_name', 'source_image_file', 'poss_ident1', 'poss_ident2', 'poss_ident3', \
                # Note that the field name, "face_thumbnail", must have _thumbnail appended
                # for admin_thumbnails to do its magic.
                    'poss_ident4', 'poss_ident5', 'face_thumbnail_thumbnail')
        }),

        # A few advanced options that we generally don't need to see.
        ('Advanced options', {
            'classes': ('collapse',),
            'fields': ('box_top','box_bottom', 'box_left', 'box_right', 'face_encoding'),
        }),
    )


admin.site.register(Face, FaceAdmin)
admin.site.register(Person)
