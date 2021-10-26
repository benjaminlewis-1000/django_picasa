#! /bin/bash

rm /locks/adding.lock
rm /locks/classify.lock
rm /locks/face_add.lock

for process in `ps uax | grep celery | grep -v grep  | awk '{print $2}'`; do
	echo $process
	kill -9 $process
done

celery -A picasa beat -l INFO --pidfile="/locks/celerybeat.pid"  &
# celery -A picasa worker -l INFO & # --uid=nobody --gid=nogroup &
celery -A picasa worker -l INFO  -c 10 --max-tasks-per-child 10 -n worker1 & # --uid=nobody --gid=nogroup &
celery -A picasa worker -l INFO  -c 10 --max-tasks-per-child 10 -n worker2 & # --uid=nobody --gid=nogroup &
celery -A picasa worker -l INFO  -c 10 --max-tasks-per-child 10 -n worker3 & # --uid=nobody --gid=nogroup &
celery -A picasa worker -l INFO  -c 10 --max-tasks-per-child 10 -n worker4 & # --uid=nobody --gid=nogroup &

