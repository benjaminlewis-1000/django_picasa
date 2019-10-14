from django.contrib import admin

# Register your models here.

from .models import ImageFile, Directory, Face

admin.site.register(ImageFile)
admin.site.register(Directory)
admin.site.register(Face)

