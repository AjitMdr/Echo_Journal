from celery import shared_task
from accounts.services.streak_checker import check_streaks
import logging
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


@shared_task(
    name='accounts.tasks.check_streaks_task',
    bind=True,
    max_retries=3,
    default_retry_delay=300,  # 5 minutes
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,  # 10 minutes
    retry_jitter=True
)
def check_streaks_task(self):
    """
    Celery task to check and update user streaks.
    This task will be scheduled to run daily at midnight.
    """
    try:
        logger.info("Starting Celery task for streak check")
        check_streaks()
        logger.info("Successfully completed streak check task")
    except Exception as exc:
        logger.error(f"Error in streak check task: {str(exc)}")
        # Retry the task in case of failure
        self.retry(exc=exc)
