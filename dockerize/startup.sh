#! /bin/bash

sleep 5 


python /code/manage.py makemigrations
python /code/manage.py makemigrations filepopulator
python /code/manage.py makemigrations face_manager
python /code/manage.py migrate
python /code/manage.py shell < /code/dockerize/make_superuser_once.py
cat <(echo "yes") - | python /code/manage.py collectstatic

mkdir /locks

rm -f /locks/celerybeat.pid
rm -f /locks/adding.lock
rm -f /locks/classify.lock
rm -f /locks/face_add.lock
rm -f /code/celerybeat-schedule.db

mkdir -p /var/run/celery /var/log/celery
chown -R nobody:nogroup /var/run/celery /var/log/celery
#chmod 777 -R /var/log/picasa
#chmod 777 -R /locks
#chmod 777 -R /media

for process in `ps uax | grep celery | grep -v grep  | awk '{print $2}'`; do
    echo $process
    kill -9 $process
done

sleep 10

# celery flower -A picasa --port=5555 &
celery -A picasa beat -l INFO --pidfile="/locks/celerybeat.pid"  &

for i in {1..8}; do
#     celery -A picasa worker -l INFO -c 4 --max-tasks-per-child 3 -n worker${i}  --uid=1001 --gid=1001 &
     celery -A picasa worker -l INFO -c 4 --max-tasks-per-child 3 -n worker${i}  --uid=root --gid=root &
done

gunicorn -b 0.0.0.0:8000 picasa.wsgi & 




while true; do 
    sleep 10
done 

