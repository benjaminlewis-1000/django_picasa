FROM debian

WORKDIR /code

RUN apt update

# Add our requirements.txt
ADD requirements.txt /code/requirements.txt

RUN apt install cmake -y
RUN apt install python3-pip -y
RUN pip3 install -r /code/requirements.txt

RUN apt remove cmake -y

RUN cp /usr/bin/python3 /usr/bin/python

RUN rm -r /var/lib/apt/lists

