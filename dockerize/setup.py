#! /usr/bin/env python

import os
import shutil
import getpass
import random
import string
import pathlib

project_path = os.path.abspath(os.path.join(__file__,"../.."))
script_path  = os.path.abspath(os.path.join(__file__,".."))

shutil.copy(os.path.join(project_path, 'requirements.txt'), script_path)

# Make the .env file

photo_location = input("What is the root directory for your photos?  ")
while not os.path.isdir(photo_location):
	photo_location = input("Location {} is not a valid directory. What is the root directory for your photos?  ".format(photo_location))

choice = 'N'
while choice.lower() == 'n':
	media_storage_dir = input("Where would you like to store files created by this app?  ")
	choice = input(f"Are you sure you'd like to put your media in {media_storage_dir}? [Y/n]  ")

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

db_dir = 'database'
thumbs_dir = 'media'
pathlib.Path(os.path.join(media_storage_dir, db_dir)).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(media_storage_dir, thumbs_dir)).mkdir(parents=True, exist_ok=True)

SECRET_KEY = ''.join([random.SystemRandom().choice("{}{}{}".\
	format(string.ascii_letters, string.digits, string.punctuation)) for i in range(50)])
DB_PWD = ''.join([random.SystemRandom().choice("{}{}{}".\
	format(string.ascii_letters, string.digits, string.punctuation)) for i in range(20)])

out_file_path = os.path.join(script_path, '.env_picasa')

with open(out_file_path, 'w') as fh:
	fh.write(f'PHOTO_ROOT={photo_location}\n')
	fh.write(f'MEDIA_ROOT={os.path.join(media_storage_dir, thumbs_dir)}\n')
	fh.write("DOMAIN_NAME=exploretheworld.tech\n")
	fh.write("MEDIA_DOMAIN=picasamedia.exploretheworld.tech\n")
	fh.write("WEBAPP_DOMAIN=picasa.exploretheworld.tech\n")
	fh.write(f"ADMIN_USERNAME={username}\n")
	fh.write(f"ADMIN_PW={pw1}\n")
	fh.write(f"DJANGO_SECRET_KEY={SECRET_KEY}\n")
	fh.write(f"DB_USER={username}\n")
	fh.write(f"DB_PWD={DB_PWD}\n")
	fh.write("DB_NAME=picasa")
