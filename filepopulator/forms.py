
from django import forms
from django.conf import settings
from django.forms import ModelForm
from .models import ImageFile, Directory

# class ImageFileForm(forms.Form):
#    filename = forms.FilePathField(path=settings.FILEPOPULATOR_FILE_PATH_FIELD_DIRECTORY, match="\.[j|J][p|P][e|E][g|G]$", max_length=255)
#    width = forms.integerField()
#    height = forms.integerField()

class ImageFileForm(ModelForm):
    class Meta:
        model = ImageFile
        fields = ['filename', 'directory', 'width', 'height']

class DirectoryForm(ModelForm):
    class Meta:
        model = Directory
        fields = ['dir_path']
