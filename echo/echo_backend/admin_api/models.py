from django.db import models
from django.utils import timezone
from django.contrib.auth import get_user_model

User = get_user_model()


class AdminLog(models.Model):
    """Model to track admin actions"""
    ACTION_TYPES = [
        ('CREATE', 'Create'),
        ('UPDATE', 'Update'),
        ('DELETE', 'Delete'),
        ('LOGIN', 'Login'),
        ('OTHER', 'Other'),
    ]

    admin_user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='admin_logs')
    action_type = models.CharField(max_length=10, choices=ACTION_TYPES)
    action_detail = models.TextField()
    # e.g., 'User', 'Journal', etc.
    target_model = models.CharField(max_length=50)
    # ID of the affected record
    target_id = models.CharField(max_length=50, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.admin_user.username} - {self.action_type} - {self.created_at}"


class DashboardMetric(models.Model):
    """Model to cache dashboard metrics"""
    metric_name = models.CharField(max_length=50, unique=True)
    metric_value = models.JSONField()  # Stores the actual metric data
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.metric_name} - {self.last_updated}"

    @classmethod
    def update_metric(cls, name, value):
        """Update or create a metric"""
        metric, _ = cls.objects.update_or_create(
            metric_name=name,
            defaults={'metric_value': value, 'last_updated': timezone.now()}
        )
        return metric
