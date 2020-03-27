from django.contrib import admin

# Register your models here.

from .models import Face, Person

admin.site.register(Face)
admin.site.register(Person)