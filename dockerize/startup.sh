#! /bin/bash

sleep 5 

python /code/manage.py makemigrations
python /code/manage.py makemigrations filepopulator
python /code/manage.py makemigrations face_manager
python /code/manage.py migrate
python /code/manage.py shell < /code/dockerize/make_superuser_once.py
cat <(echo "yes") - | python /code/manage.py collectstatic

mkdir /locks

rm /locks/celerybeat.pid
rm /locks/adding.lock
rm /locks/face_add.lock

mkdir -p /var/run/celery /var/log/celery
chown -R nobody:nogroup /var/run/celery /var/log/celery
#chmod 777 -R /var/log/picasa
#chmod 777 -R /locks
#chmod 777 -R /media

# celery flower -A picasa --port=5555 &
celery -A picasa beat -l INFO --pidfile="/locks/celerybeat.pid"  &
celery -A picasa worker -l INFO & # --uid=nobody --gid=nogroup &
gunicorn -b 0.0.0.0:8000 picasa.wsgi & 

while true; do 
    sleep 10
done 

