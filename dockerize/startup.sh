#! /bin/bash

sleep 5 


# python /code/manage.py makemigrations
# python /code/manage.py makemigrations filepopulator
# python /code/manage.py makemigrations face_manager
# python /code/manage.py migrate
# python /code/manage.py shell < /code/dockerize/make_superuser_once.py
# cat <(echo "yes") - | python /code/manage.py collectstatic

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

# for i in {1..8}; do
#     celery -A picasa worker -l INFO -c 4 --max-tasks-per-child 3 -n worker${i}  --uid=1001 --gid=1001 &
#     celery -A picasa worker -l INFO -c 4 --max-tasks-per-child 3 -n worker${i}  --uid=root --gid=root &
# done
celery -A picasa worker -l INFO -c 20 --max-tasks-per-child 3 -n worker  & # --uid=nobody --gid=nogroup &

# Previously ran with gunicorn's defaults (1 sync worker, 1 thread), which
# serializes *every* request - including every single face-thumbnail image
# request a gallery page fires - through one process, regardless of how many
# CPU threads the host actually has. 4 workers x 8 threads (gthread) = 32
# concurrent request slots, a middle ground that fixes that serialization
# without competing too hard against celery's own -c 20 concurrency above
# for the same host's cores.
#
# --timeout 600 (gunicorn's own default is 30s): the file-upload endpoint
# (api/upload_views.py) accepts uploads up to UPLOAD_MAX_FILE_SIZE_BYTES
# (4GiB by default) -- gunicorn's timeout kills a worker that hasn't
# checked in within this window, and a large upload over a modest home
# connection can easily take several minutes just to receive the request
# body, well past the 30s default. 10 minutes is a starting point, not a
# hard ceiling verified against real large-file uploads yet.
gunicorn -b 0.0.0.0:8000 --workers 4 --threads 8 --worker-class gthread --timeout 600 picasa.wsgi &




while true; do 
    sleep 10
done 

