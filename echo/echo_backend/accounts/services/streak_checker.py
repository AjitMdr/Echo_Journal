from django.utils import timezone
from datetime import timedelta
from accounts.models import Streak
from journal.models import Journal
import logging

logger = logging.getLogger(__name__)


def check_streaks():
    """
    Check and update user streaks daily.
    This function should be called by a scheduled task (e.g., cron job).
    """
    try:
        # Get current time in UTC
        now = timezone.now()
        today = now.date()
        yesterday = today - timedelta(days=1)

        logger.info(f"Starting streak check for {today}")

        # Get all users with streaks
        streaks = Streak.objects.all()
        logger.info(f"Found {streaks.count()} users with streaks")

        for streak in streaks:
            try:
                if not streak.is_streak_current():
                    # If streak is not current (more than 1 day has passed), reset it
                    if streak.current_streak > 0:  # Only log if there was a streak
                        logger.info(
                            f'Streak broken for user {streak.user.username} - {streak.current_streak} days')
                    streak.current_streak = 0  # Reset to 0 for broken streaks
                    streak.save()
                    continue

                # Check if user has a journal entry for today
                has_today_entry = Journal.objects.filter(
                    user=streak.user,
                    date__date=today,
                    is_deleted=False
                ).exists()

                # Check if user has a journal entry for yesterday
                has_yesterday_entry = Journal.objects.filter(
                    user=streak.user,
                    date__date=yesterday,
                    is_deleted=False
                ).exists()

                if not has_today_entry and not has_yesterday_entry:
                    # User hasn't journaled in 2 days, streak is broken
                    if streak.current_streak > 0:  # Only log if there was a streak
                        logger.info(
                            f'Streak broken for user {streak.user.username} - {streak.current_streak} days')
                    streak.current_streak = 0  # Reset to 0 for broken streaks
                    streak.save()
                elif has_today_entry:
                    # User has an entry for today, streak continues or starts
                    if streak.current_streak == 0:
                        streak.current_streak = 1
                    else:
                        streak.current_streak += 1

                    if streak.current_streak > streak.longest_streak:
                        streak.longest_streak = streak.current_streak
                    streak.save()
                    logger.info(
                        f'Streak updated for user {streak.user.username}: {streak.current_streak} days')

            except Exception as e:
                logger.error(
                    f'Error processing streak for user {streak.user.username}: {str(e)}')
                continue

        logger.info("Streak check completed successfully")
    except Exception as e:
        logger.error(f"Error in streak check: {str(e)}")
        raise
