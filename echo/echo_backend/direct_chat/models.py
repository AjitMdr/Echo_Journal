from django.db import models
from django.conf import settings
from django.utils import timezone


class Conversation(models.Model):
    participants = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        related_name='conversations'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"Conversation {self.id} between {', '.join([user.username for user in self.participants.all()])}"

    def get_messages(self):
        """Get all messages between the participants"""
        return self.messages.all().order_by('timestamp')

    def get_last_message(self):
        """Get the last message in the conversation"""
        return self.get_messages().last()

    def mark_messages_as_read(self, user):
        """Mark all messages sent to the user as read"""
        self.messages.filter(
            receiver=user,
            is_read=False
        ).update(is_read=True)

    def get_unread_count(self, user):
        """Get the count of unread messages for a user"""
        return self.messages.filter(
            receiver=user,
            is_read=False
        ).count()


class DirectMessage(models.Model):
    sender = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='sent_messages',
        on_delete=models.CASCADE
    )
    receiver = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='received_messages',
        on_delete=models.CASCADE
    )
    conversation = models.ForeignKey(
        Conversation,
        related_name='messages',
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    content = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)
    is_read = models.BooleanField(default=False)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f'{self.sender.username} to {self.receiver.username}: {self.content[:20]}'
