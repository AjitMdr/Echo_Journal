import os
from celery import Celery
from celery.schedules import crontab
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

app = Celery('echo_backend')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

# Configure the periodic tasks
app.conf.beat_schedule = {
    'check-user-streaks': {
        'task': 'accounts.tasks.check_streaks_task',
        'schedule': crontab(hour=0, minute=0),  # Run at midnight every day
    },
}


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
