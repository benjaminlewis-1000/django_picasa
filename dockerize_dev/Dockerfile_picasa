FROM debian

WORKDIR /code

RUN apt update

RUN apt install cmake -y
RUN apt install python3-pip -y

# Add our requirements.txt
RUN pip3 install face_recognition
ADD requirements.txt /code/requirements.txt
RUN pip3 install -r /code/requirements.txt

RUN apt remove cmake python3-pip -y

RUN cp /usr/bin/python3 /usr/bin/python

RUN apt-get install -y wget

RUN rm -r /var/lib/apt/lists

ADD startup.sh /startup.sh
