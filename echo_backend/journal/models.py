from django.db import models
from django.conf import settings


class JournalManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(is_deleted=False)


class Journal(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    language = models.CharField(max_length=20, default='en')
    date = models.DateTimeField(auto_now_add=True)
    edit_date = models.DateTimeField(auto_now=True)
    is_deleted = models.BooleanField(default=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='journals',
        db_column='user_id'  # Explicitly set the column name
    )

    # Default manager that will filter out deleted items
    objects = JournalManager()
    # Manager to access all objects including deleted ones
    all_objects = models.Manager()

    def __str__(self):
        return f'{self.title} - {self.user.username}'

    def delete(self, *args, **kwargs):
        """Soft delete the journal entry"""
        self.is_deleted = True
        self.save()

    def hard_delete(self, *args, **kwargs):
        """Permanently delete the journal entry"""
        super().delete(*args, **kwargs)

    def restore(self):
        """Restore a soft-deleted journal entry"""
        self.is_deleted = False
        self.save()

    class Meta:
        db_table = 'journal_journal'  # Explicitly set the table name
