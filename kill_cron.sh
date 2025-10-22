#! /bin/bash

rm /locks/adding.lock
rm /locks/classify.lock
rm /locks/face_add.lock
rm /code/celerybeat-schedule.db

for process in `ps uax | grep celery | grep -v grep  | awk '{print $2}'`; do
	echo $process
	kill -9 $process
done

celery -A picasa beat -l INFO --pidfile="/locks/celerybeat.pid"  &
# celery -A picasa worker -l INFO & # --uid=nobody --gid=nogroup &

# for i in {1..6}; do
    # celery -A picasa worker -l INFO -c 4 --max-tasks-per-child 3 -n worker${i}  & # --uid=nobody --gid=nogroup &
celery -A picasa worker -l INFO -c 20 --max-tasks-per-child 3 -n worker  & # --uid=nobody --gid=nogroup &
# done

