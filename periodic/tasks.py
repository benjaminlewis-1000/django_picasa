from celery import task
from celery import shared_task
# from celery.task.schedules import crontab
from celery.utils.log import get_task_logger
import time

logger = get_task_logger(__name__)


# @task(name='summary')
# def send_import_summary():
#     pass

# @shared_task
# def send_notification():
#     print('Here I am!')

@task(name='time_write')
def task_write_time():
    '''Writes the time to a file'''
    fname = '/home/benjamin/time.txt'
    logger.info("Write time")
    with open(fname, 'a') as fh:
        fh.write('The time is : {}\n'.format( time.time() ) )
                                    
