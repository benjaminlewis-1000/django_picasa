#! /bin/bash

rm /locks/adding.lock
rm /locks/classify.lock
rm /locks/face_add.lock

for process in `ps uax | grep celery | grep -v grep  | awk '{print $2}'`; do
	echo $process
	kill -9 $process
done

celery -A picasa beat -l INFO --pidfile="/locks/celerybeat.pid"  &
celery -A picasa worker -l INFO & # --uid=nobody --gid=nogroup &
