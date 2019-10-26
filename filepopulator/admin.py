from django.contrib import admin

# Register your models here.

from .models import ImageFile, Directory

admin.site.register(ImageFile)
admin.site.register(Directory)
# admin.site.register(Face)

