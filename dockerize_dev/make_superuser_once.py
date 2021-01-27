from django.contrib.auth.models import User
import os
from django.contrib.auth import get_user_model

userList =User.objects.values()

for user in userList:
    print(user)
    if user['username'] == os.environ['ADMIN_USERNAME']:
        print("no action needed")
        exit()


User = get_user_model()
admin_username = os.environ['ADMIN_USERNAME']
admin_pass = os.environ['ADMIN_PW']
admin_email = os.environ['ADMIN_EMAIL']
User.objects.create_superuser(admin_username, admin_email, admin_pass)
