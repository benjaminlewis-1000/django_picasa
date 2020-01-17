#! /bin/bash

python /code/manage.py makemigrations
python /code/manage.py makemigrations filepopulator
python /code/manage.py migrate


celery flower -A picasa --port=5555 &
celery -A picasa beat -l info --pidfile="/var/run/lock/celerybeat.pid"  &
celery -A picasa worker -l info &
python /code/manage.py runserver 

