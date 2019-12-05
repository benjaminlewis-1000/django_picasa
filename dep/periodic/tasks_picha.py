from celery.decorators import periodic_task
from celery.task.schedules import crontab
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

@periodic_task(
    run_every=(crontab(minute='*')),
    name="task_write_time",
    ignore_result=True
)
def task_write_time():
    '''Writes the time to a file'''
    fname = '/home/benjamin/time.txt'
    logger.info("Write time")
    with open(fname, 'a') as fh:
        fh.write('The time is : {}\n'.format( time.time() ) )
