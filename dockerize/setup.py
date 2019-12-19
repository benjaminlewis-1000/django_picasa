#! /usr/bin/env python

import getpass
import os
import pathlib
import random
import shutil
import string
import xmltodict
import subprocess

project_path = os.path.abspath(os.path.join(__file__,"../.."))
script_path  = os.path.abspath(os.path.join(__file__,".."))

print("TODO: Test file location")
print("TODO: explicit ports for postgres and redis")

param_file = os.path.join(project_path, 'image_face_extractor', 'parameters.xml')
if not os.path.isdir(os.path.dirname(param_file)):
	raise ValueError("Need to know the path to the face extraction code to fetch the parameter file.")
if not os.path.isfile(param_file):
	raise ValueError("Need to know the path to the face extraction code parameter file.")

with open(param_file) as pf:
	params = xmltodict.parse(pf.read())

client_port_detect = params['params']['ports']['client_return_port']
print(client_port_detect)

shutil.copy(os.path.join(project_path, 'requirements.txt'), script_path)

# Make the .env file

photo_location = input("What is the root directory for your photos?  ")
while not os.path.isdir(photo_location):
	photo_location = input("Location {} is not a valid directory. What is the root directory for your photos?  ".format(photo_location))

choice = 'N'
while choice.lower() == 'n':
	media_storage_dir = input("Where would you like to store files created by this app?  ")
	choice = input(f"Are you sure you'd like to put your media in {media_storage_dir}? [Y/n]  ")

test_dir = input("Where are your test files?  ")
while not os.path.isdir(test_dir):
    test_dir = input(f"Location {test_dir} is not a valid directory. What is the location of your test files? ")

choice = 'N'
while choice.lower() == 'n':
	username = input("What would you like your username to be for the admin console? ")
	choice = input(f"Are you sure you want your username to be {username}? [Y/n] ")

pw1 = ''
pw2 = 'ghjk'
while pw1 != pw2:
	if pw1 == '':
		pw1 = getpass.getpass(f"Please input a password for user {username}: ")
	else:
		pw1 = getpass.getpass(f"Passwords don't match! Please input a password for user {username}: ")
	pw2 = getpass.getpass("Please put in your password again: ")

if not os.path.isdir(media_storage_dir):
	pathlib.Path(media_storage_dir).mkdir(parents=True, exist_ok=True)

db_dir = os.path.join(media_storage_dir, 'database')
thumbs_dir = os.path.join(media_storage_dir, 'files')
db_name = 'picasa'
pathlib.Path( thumbs_dir).mkdir(parents=True, exist_ok=True)

# Remove the database directory. 
if os.path.isdir(db_dir):
	rmd = input("This script needs to remove the database directory to get the initialization to work properly. OK? (y/N) ")
	if rmd.lower() != 'y':
		print("Not removing the database directory. You will need to initialize the database table manually.")
	else:
		rmd = input("SECOND CHANCE!!! Are you sure? (y/N) ")
		if rmd.lower() == 'y':
			print("removing...")
			os.system(f"sudo rm -r {db_dir}")
		else:
			print("Not removing the database directory. You will need to initialize the database table manually.")
pathlib.Path( db_dir ).mkdir(parents=True, exist_ok=True)

SECRET_KEY = ''.join([random.SystemRandom().choice("{}{}{}".\
	format(string.ascii_letters, string.digits, string.punctuation)) for i in range(50)])
DB_PWD = ''.join([random.SystemRandom().choice("{}{}{}".\
	format(string.ascii_letters, string.digits, string.punctuation)) for i in range(20)])

out_file_path = os.path.join(script_path, '.env')

with open(out_file_path, 'w') as fh:
	fh.write(f"ADMIN_PW={pw1}\n")
	fh.write(f"ADMIN_USERNAME={username}\n")
	fh.write(f"CLIENT_FACE_PORT={client_port_detect}\n")
	fh.write(f"DB_FILES_LOCATION={db_dir}\n")
	fh.write(f"DB_NAME={db_name}\n")
	fh.write(f"DB_PWD={DB_PWD}\n")
	fh.write(f"DB_USER={username}\n")
	fh.write(f"DJANGO_SECRET_KEY={SECRET_KEY}\n")
	fh.write(f"DOMAIN_NAME=exploretheworld.tech\n")
	fh.write(f"MEDIA_DOMAIN=picasamedia.exploretheworld.tech\n")
	fh.write(f"MEDIA_FILES_LOCATION={thumbs_dir}\n")
	fh.write(f"PHOTO_ROOT={photo_location}\n")
	fh.write(f"WEBAPP_DOMAIN=picasa.exploretheworld.tech\n")
	fh.write(f"DJANGO_FILES_ROOT={project_path}\n")
	fh.write(f"IN_DOCKER=True\n")
	fh.write(f"TEST_PHOTOS={test_dir}\n")

PSQL_SCRIPT = '''
#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER {username} PASSWORD '{pwd}';
    ALTER USER {username} CREATEDB;
    CREATE DATABASE {db_name};
    GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {username};
EOSQL
'''.format(username = username, pwd = DB_PWD, db_name = db_name)

with open('psql_init.sh', 'w') as ph:
	ph.write(PSQL_SCRIPT)


# os.system(f"docker exec db_picasa su postgres -c psql \"CREATE USER {username} PASSWORD '{DB_PWD}'; \"")
# os.system(f"docker exec db_picasa su postgres -c psql \"ALTER USER {username} CREATEDB; \"")
# os.system(f"docker exec db_picasa su postgres -c psql \"CREATE DATABASE picasa PASSWORD '{DB_PWD}'; \"")

os.system("docker-compose build")

process = subprocess.Popen("docker-compose up", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# process.terminate()
