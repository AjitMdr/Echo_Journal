# models.py
from django.db import models
from django.core.validators import EmailValidator
from django.contrib.auth.models import AbstractUser

from django.utils import timezone
from datetime import datetime, timedelta


class User(AbstractUser):
    ROLE_CHOICES = [
        ('USER', 'Regular User'),
        ('ADMIN', 'Administrator'),
        ('SUPERADMIN', 'Super Administrator'),
    ]

    email = models.EmailField(
        unique=True,
        validators=[EmailValidator(message="Enter a valid email address.")],
        error_messages={"unique": "A user with this email already exists."}
    )
    username = models.CharField(
        max_length=255,
        unique=True,
        error_messages={"unique": "A user with this username already exists."}
    )
    phone_number = models.CharField(
        max_length=15, blank=True, null=True, unique=True)

    profile_picture = models.ImageField(
        upload_to='profile_pictures/', null=True, blank=True)

    is_verified = models.BooleanField(default=False)
    two_factor_enabled = models.BooleanField(default=False)
    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default='USER'
    )

    groups = models.ManyToManyField(
        'auth.Group',
        related_name='custom_user_groups',
        blank=True,
        help_text='The groups this user belongs to.',
        related_query_name='user'
    )

    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='custom_user_permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_query_name='user'
    )

    REQUIRED_FIELDS = ['username']
    USERNAME_FIELD = 'email'

    def __str__(self):
        return self.username

    def save(self, *args, **kwargs):
        # Automatically set role based on is_staff and is_superuser
        if self.is_superuser:
            self.role = 'SUPERADMIN'
        elif self.is_staff:
            self.role = 'ADMIN'
        super().save(*args, **kwargs)

    @property
    def friends(self):
        """Get all friends of this user through friendships"""
        friend_ids = self.friendships_as_user.values_list(
            'friend_id', flat=True)
        friend_ids = list(
            friend_ids) + list(self.friendships_as_friend.values_list('user_id', flat=True))
        return User.objects.filter(id__in=friend_ids)

    class Meta:
        db_table = 'accounts_user'  # Explicitly set the table name


class Streak(models.Model):
    user = models.OneToOneField(
        User, on_delete=models.CASCADE, related_name='streak')
    current_streak = models.IntegerField(default=0)
    longest_streak = models.IntegerField(default=0)
    last_journal_date = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def is_streak_current(self):
        """Check if the streak is still current (not broken)"""
        if not self.last_journal_date:
            return False

        now = timezone.now()
        today = now.date()
        last_date = self.last_journal_date.date()
        days_since_last = (today - last_date).days

        # If the last entry was today or yesterday, streak is current
        return days_since_last <= 1

    def update_streak(self):
        today = timezone.now().date()

        if self.last_journal_date:
            last_date = self.last_journal_date.date()
            days_since_last = (today - last_date).days

            if today == last_date:
                return  # Already updated today
            elif days_since_last == 1:
                # Streak continues
                self.current_streak += 1
                if self.current_streak > self.longest_streak:
                    self.longest_streak = self.current_streak
            else:
                # Streak broken - more than one day has passed
                self.current_streak = 0  # Reset to 0 for broken streaks
        else:
            # First journal entry
            self.current_streak = 1

        self.last_journal_date = timezone.now()
        self.save()

    def __str__(self):
        return f"{self.user.username}'s Streak: {self.current_streak} days"


class Badge(models.Model):
    BADGE_TYPES = [
        ('STREAK', 'Streak Badge'),
        ('JOURNAL', 'Journal Badge'),
        ('MOOD', 'Mood Badge'),
    ]

    name = models.CharField(max_length=100)
    description = models.TextField()
    badge_type = models.CharField(max_length=20, choices=BADGE_TYPES)
    icon = models.CharField(max_length=50)  # Emoji or icon identifier
    requirement = models.IntegerField()  # Required streak count or other metric
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.badge_type})"


class UserBadge(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='badges')
    badge = models.ForeignKey(Badge, on_delete=models.CASCADE)
    earned_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'badge')

    def __str__(self):
        return f"{self.user.username} - {self.badge.name}"


class Friendship(models.Model):
    user = models.ForeignKey(
        User, related_name='friendships', on_delete=models.CASCADE)
    friend = models.ForeignKey(
        User, related_name='friend_of', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['user', 'friend']
        indexes = [
            models.Index(fields=['user']),
            models.Index(fields=['friend']),
        ]

    def __str__(self):
        return f"{self.user.username} is friends with {self.friend.username}"


# Add a property to User model to easily access friends
User.add_to_class('friends', property(lambda self: User.objects.filter(
    id__in=self.friendships.values_list('friend_id', flat=True)
)))
