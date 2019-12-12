FROM debian

WORKDIR /code

RUN apt update
RUN apt install git -y

# Get the code for picasa, so we can run requirements.txt
RUN git clone --recurse-submodules https://github.com/benjaminlewis-1000/filepopulator-django picasa
# WORKDIR /code/
ADD requirements.txt /code/requirements.txt

RUN apt install cmake -y
RUN apt install python3-pip -y
# Get 
# RUN pip install dlib
RUN pip3 install -r /code/requirements.txt

RUN apt install redis -y

RUN apt install postgresql -y
RUN apt install postgresql-contrib -y

RUN which psql

ARG DB_NAME=picasa
ARG DB_USER=benjamin
ARG DB_PASS=lewis
ARG USER_EMAIL=benjamin.lewis.1000@gmail.com
ARG TMP_PASS=tmp

# export DB_NAME=picasa
# export DB_USER=benjamin
# export DB_PASS=lewis



RUN apt remove cmake -y

RUN cp /usr/bin/python3 /usr/bin/python

# RUN apt install nvidia-cuda-toolkit

RUN rm -r /var/lib/apt/lists


# # Actually, we'll want this all to be run via docker compose, but good try?!?
# RUN service postgresql start && \
# 	su - postgres -c "psql -c \"CREATE USER ${DB_USER} PASSWORD '${DB_PASS}'; \" " && \
# 	su - postgres -c "psql -c \"ALTER USER ${DB_USER} CREATEDB; \" " && \
# 	su - postgres -c "psql -c \"CREATE DATABASE ${DB_NAME} OWNER ${DB_USER}\" "

# RUN service postgresql start &&  \
# 	python manage.py makemigrations && \
# 	python manage.py migrate && \ 
# 	echo "from django.contrib.auth.models import User; User.objects.create_superuser('${DB_USER}', '${USER_EMAIL}', '${TMP_PASS}')" | python manage.py shell
