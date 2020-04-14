"""
Django settings for picasa project.

Generated by 'django-admin startproject' using Django 2.2.4.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""

import os
from celery.schedules import crontab
import logging
from unipath import Path

# # Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Static files (CSS, JavaScript, Images)
# # https://docs.djangoproject.com/en/2.2/howto/static-files/

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if 'IN_DOCKER' in os.environ.keys():
    in_docker = True
else:
    in_docker = False

production = False
if 'PRODUCTION' in os.environ.keys():
    if os.environ['PRODUCTION'] == 'True':
        production = True

if in_docker:
    SECRET_KEY = os.environ['DJANGO_SECRET_KEY']
    DB_NAME = os.environ['DB_NAME']
    DB_USER = os.environ['DB_USER']
    DB_PASS = os.environ['DB_PWD']
    MEDIA_ROOT = '/media' # os.environ['MEDIA_FILES_LOCATION']
    if production:
        REDIS_HOST = 'task_redis'
        DB_HOST = 'db_django'
    else:
        REDIS_HOST = 'task_redis_dev'
        DB_HOST = 'db_django_dev'
    PHOTO_ROOT = '/photos'
    TEST_IMG_DIR_FILEPOPULATE = '/test_imgs_filepopulate'
#    ALLOWED_HOSTS = ['localhost', '127.0.0.1', os.environ['WEBAPP_DOMAIN']]
    ALLOWED_HOSTS = ['localhost', '127.0.0.1', os.environ['DOMAINNAME'], os.environ['WEBAPP_DOMAIN'], 'flower.' + os.environ['DOMAINNAME']]
#    ALLOWED_HOSTS = ['localhost', '127.0.0.1', os.environ['DOMAINNAME'], os.environ['WEBAPP_DOMAIN'] ]
#    STATIC_URL = 'http://localhost/static/'
#    MEDIA_URL  = 'http://localhost:8080/'
    STATIC_URL = 'https://' + os.environ['MEDIA_DOMAIN'] + '/static/'
    MEDIA_URL = 'https://' + os.environ['MEDIA_DOMAIN'] + '/media/'
    LOG_DIR = '/var/log/picasa'
    STATIC_ROOT = os.environ['STATIC_LOCATION']
    MEDIA_URL_USER = os.environ['APACHE_USER']
    MEDIA_URL_PW = os.environ['APACHE_PWD']
else:
    DB_NAME = 'picasa'
    DB_USER = 'benjamin'
    DB_PASS = 'lewis'
    DB_HOST = 'localhost'
    REDIS_HOST = 'localhost'
    MEDIA_ROOT = os.path.join(PROJECT_ROOT, 'media')
    SECRET_KEY = '46r=!b*lkf2-^#a=40kq(9nxkfz53d7!ft7_iccd1!2aa%q@6z'
    ALLOWED_HOSTS = ['localhost', '127.0.0.1']
    PHOTO_ROOT = '/home/benjamin/git_repos/picasa_files/actual_imgs'
    TEST_IMG_DIR_FILEPOPULATE = '/home/benjamin/git_repos/picasa_files/test_imgs'

    STATIC_URL = 'http://localhost/static/'
    MEDIA_URL  = 'http://localhost/media/'
    STATIC_SERVER = '/var/www/html/static'
    LOG_DIR = '/home/benjamin/git_repos/picasa_files/logs'

# Set up logging
formatter = logging.Formatter('%(levelname)s | %(asctime)s - %(message)s')
LOGGER = logging.getLogger("__main__")
LOGGER.setLevel(logging.DEBUG)
LOGGER.propagate = False

fh_low = logging.FileHandler(os.path.join(LOG_DIR, 'picasa_debug.log'), 'a')
fh_low.setLevel(logging.DEBUG)
fh_low.setFormatter(formatter)
LOGGER.addHandler(fh_low)

fh_high = logging.FileHandler(os.path.join(LOG_DIR, 'picasa_important.log'), 'a')
fh_high.setLevel(logging.WARNING)
fh_high.setFormatter(formatter)
LOGGER.addHandler(fh_high)

fh_mid = logging.FileHandler(os.path.join(LOG_DIR, 'picasa_lite_debug.log'), 'a')
fh_mid.setLevel(logging.INFO)
fh_mid.setFormatter(formatter)
LOGGER.addHandler(fh_mid)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

if in_docker:
    LOGGER.warning("Allowed hosts is wrong in docker")

LOGGER.debug(f"Are we in production? {production}")

SITE_ROOT = os.path.dirname(os.path.realpath(__file__))

LOCKFILE = '/locks/adding.lock' # path.join(PROJECT_ROOT, 'adding.lock')
FACE_LOCKFILE = '/locks/face_add.lock'
# if os.path.isfile(LOCKFILE):
#     os.remove(LOCKFILE)

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = production # False

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # 'periodic.apps.PeriodicConfig',
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'filepopulator',
    'face_manager',
    'api',
    # 'upload_img',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

CORS_ORIGIN_WHITELIST = [
    "https://192.168.1.15:8080",
    "https://192.168.1.145:8080",
    "https://facewire.exploretheworld.tech"
]

CORS_ORIGIN_REGEX_WHITELIST = [
    r"^https://192.168.1.*",
    r"^http://192.168.1.*",
]

CORS_ORIGIN_ALLOW_ALL=True
LOGGER.critical("CORS allowing from all...")

ROOT_URLCONF = 'picasa.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'picasa.wsgi.application'


# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': DB_NAME,
        'USER': DB_USER,
        'PASSWORD': DB_PASS,
        'HOST': DB_HOST,
        'PORT': '5432',
    }
}

# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Token authentication: https://simpleisbetterthancomplex.com/tutorial/2018/11/22/how-to-implement-token-authentication-using-django-rest-framework.html
REST_FRAMEWORK = {
     'DEFAULT_AUTHENTICATION_CLASSES': [
        # 'rest_framework.authentication.BasicAuthentication',
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',  # <-- And here
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.LimitOffsetPagination',
    'PAGE_SIZE': 100, 
}


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'America/New_York'
LOGGER.debug(f"Time zone is {TIME_ZONE}")

USE_I18N = True

USE_L10N = True

USE_TZ = True

# CELERY STUFF
CELERY_BROKER_URL = f'redis://{REDIS_HOST}:6379'
CELERY_RESULT_BACKEND = f'redis://{REDIS_HOST}:6379'
CELERY_ACCEPT_CONTENT = ['application/json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE


if production: 
    CELERY_BEAT_SCHEDULE = {
        'filepopulate_root': {
            'task': 'filepopulator.populate_files_from_root',
            'schedule': crontab(minute='*/30') # , hour='*/6'),
            # 'schedule': crontab(minute='*/10'),
        },
        'dirs_datetimes': {
            'task': 'filepopulator.update_dir_dates',
            'schedule': crontab(minute=0, hour='*/12')
        },
        'face_add': {
            'task': 'face_manager.face_extraction', 
            'schedule': crontab( minute = '*/1'),
            # 'schedule': crontab( minute = '0', hour='*/2'),
            'options': {
                'expires': 30,
                # 'expires': 1800,
            }
        }

    }

else:
    CELERY_BEAT_SCHEDULE = {
    
        'filepopulate_root': {
            'task': 'filepopulator.populate_files_from_root',
            'schedule': crontab(minute='*/1'),
        },
        'dirs_datetimes': {
            'task': 'filepopulator.update_dir_dates',
            'schedule': crontab(minute='*/1')
        },
        'face_add': {
            'task': 'face_manager.face_extraction', 
            'schedule': crontab( minute = '*/1'),
            # 'schedule': crontab( minute = '0', hour='*/2'),
            'options': {
                'expires': 30,
                # 'expires': 1800,
            }
        }

    }

if not 'filepopulate_root' in CELERY_BEAT_SCHEDULE.keys():
    LOGGER.error("Filepopulate celery task not instantiated.")
    raise ValueError('Filepopulate celery task not instantiated.')
if not 'dirs_datetimes' in CELERY_BEAT_SCHEDULE.keys():
    LOGGER.error("Dirs_datetimes celery task not instantiated.")
    raise ValueError('Dirs_datetimes celery task not instantiated.')

# FILEPOPULATOR settings


# FILEPOPULATOR_THUMBNAIL_DIR = '/home/benjamin/git_repos/picasa_files/thumbnails' # place to store thumbnails that are derived from the images. 
# FILEPOPULATOR_THUMBNAIL_SIZE_TINY = (30, 30)
# FILEPOPULATOR_THUMBNAIL_SIZE = (250, 250)
FILEPOPULATOR_THUMBNAIL_SIZE_BIG = (500, 500)
FILEPOPULATOR_THUMBNAIL_SIZE_MEDIUM = (250, 250)
FILEPOPULATOR_THUMBNAIL_SIZE_SMALL = (100, 100)
FILEPOPULATOR_SERVER_IMG_DIR = PHOTO_ROOT # root location of images you want to index into. (This maybe will change)
FILEPOPULATOR_CODE_DIR = PROJECT_ROOT # '/home/benjamin/git_repos/local_picasa' # root directory of the code. 
FILEPOPULATOR_VAL_DIRECTORY = TEST_IMG_DIR_FILEPOPULATE  # point to a directory that will have validation images when testing the app.
FILEPOPULATOR_MAX_SHORT_EDGE_THUMBNAIL =150 # Maximum size of the short edge for thumbnails.

FACE_THUMBNAIL_SIZE=(200, 200)

# logging.error("TODO: Set up apache server. STATIC_URL needs to be served by it.")

BLANK_FACE_NAME='_NO_FACE_ASSIGNED_'

# Parameters for the machine learning model
FACE_NUM_THRESH=50
IGNORED_NAMES=[BLANK_FACE_NAME, '.ignore', '.realignore', '.jessicatodo', '.cutler_tbd']
