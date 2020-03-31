#! /bin/bash

sleep 5 

python /code/manage.py makemigrations
python /code/manage.py makemigrations filepopulator
python /code/manage.py makemigrations face_manager
python /code/manage.py migrate
python /code/manage.py shell < /code/dockerize/make_superuser_once.py
cat <(echo "yes") - | python /code/manage.py collectstatic

rm /var/run/lock/celerybeat.pid
rm /code/adding.lock
rm /face_add.lock

# celery flower -A picasa --port=5555 &
celery -A picasa beat -l info --pidfile="/var/run/lock/celerybeat.pid"  &
celery -A picasa worker -l info &
gunicorn -b 0.0.0.0:8000 picasa.wsgi & 
# python /code/manage.py runserver 

while true; do 
    sleep 10
done 

