from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('face_manager', '0008_face_verification_cluster_group'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='person',
            name='num_faces',
        ),
        migrations.RemoveField(
            model_name='person',
            name='num_possibilities',
        ),
        migrations.RemoveField(
            model_name='person',
            name='num_unverified_faces',
        ),
    ]
