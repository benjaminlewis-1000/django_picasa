=====
Filepopulator
=====

Filepopulator is a Django app designed to search a root directory for JPG-type files and add them to the database along with metadata. It can be used in conjunction with a face detection app (under future development) to get faces in these pictures for web processing.

Quick start
-----------

1. Add "filepopulator" to your INSTALLED_APPS setting like this::

    INSTALLED_APPS = [
        ...
        'filepopulator',
    ]

2. Include the polls URLconf in your project urls.py like this::

    path('filepopulator/', include('filepopulator.urls')),

3. Run `python manage.py migrate` to create the filepopulator models.

4. Set the following variables in your settings.py: 

FILEPOPULATOR_THUMBNAIL_DIR - place to store thumbnails that are derived from the images. 
FILEPOPULATOR_THUMBNAIL_SIZE_TINY = (30, 30)
FILEPOPULATOR_THUMBNAIL_SIZE_SMALL = (100, 100)
FILEPOPULATOR_THUMBNAIL_SIZE_MEDIUM = (250, 250)
FILEPOPULATOR_THUMBNAIL_SIZE = (250, 250)
FILEPOPULATOR_THUMBNAIL_SIZE_BIG = (500, 500)
FILEPOPULATOR_SERVER_IMG_DIR - root location of images you want to index into. (This maybe will change)
FILEPOPULATOR_CODE_DIR - root directory of the code. 
FILEPOPULATOR_VAL_DIRECTORY - point to a directory that will have validation images when testing the app.
FILEPOPULATOR_MAX_SHORT_EDGE_THUMBNAIL - Maximum size of the short edge for thumbnails.

# TODO : More about adding task schedulers, 