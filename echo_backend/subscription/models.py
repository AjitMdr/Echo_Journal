from django.db import models
from django.conf import settings
from django.utils import timezone


class Plan(models.Model):
    """Subscription plan model"""
    PLAN_TYPES = [
        ('FREE', 'Free'),
        ('BASIC', 'Basic'),
        ('PREMIUM', 'Premium'),
    ]

    name = models.CharField(max_length=100)
    plan_type = models.CharField(max_length=20, choices=PLAN_TYPES)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    duration_days = models.IntegerField()  # Duration in days
    description = models.TextField()
    features = models.JSONField()  # Store features as JSON
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - ${self.price}"

    class Meta:
        db_table = 'subscription_plan'


class Subscription(models.Model):
    """User subscription model"""
    STATUS_CHOICES = [
        ('ACTIVE', 'Active'),
        ('EXPIRED', 'Expired'),
        ('CANCELLED', 'Cancelled'),
        ('PENDING', 'Pending'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='subscriptions'
    )
    plan = models.ForeignKey(
        Plan,
        on_delete=models.PROTECT,
        related_name='subscriptions'
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='PENDING'
    )
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    is_auto_renewal = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} - {self.plan.name}"

    def is_active(self):
        """Check if subscription is currently active"""
        now = timezone.now()
        return (
            self.status == 'ACTIVE' and
            self.start_date <= now <= self.end_date
        )

    def cancel(self):
        """Cancel the subscription"""
        self.status = 'CANCELLED'
        self.is_auto_renewal = False
        self.save()

    def renew(self):
        """Renew the subscription"""
        if self.is_auto_renewal and self.status != 'CANCELLED':
            self.start_date = self.end_date
            self.end_date = self.start_date + timezone.timedelta(
                days=self.plan.duration_days
            )
            self.status = 'ACTIVE'
            self.save()

    class Meta:
        db_table = 'subscription_subscription'


class Payment(models.Model):
    """Payment model for subscriptions"""
    PAYMENT_STATUS = [
        ('PENDING', 'Pending'),
        ('SUCCESS', 'Success'),
        ('FAILED', 'Failed'),
        ('REFUNDED', 'Refunded'),
    ]

    PAYMENT_METHODS = [
        ('CARD', 'Credit/Debit Card'),
        ('PAYPAL', 'PayPal'),
        ('BANK', 'Bank Transfer'),
    ]

    subscription = models.ForeignKey(
        Subscription,
        on_delete=models.PROTECT,
        related_name='payments'
    )
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=3, default='USD')
    payment_method = models.CharField(max_length=20, choices=PAYMENT_METHODS)
    status = models.CharField(
        max_length=20,
        choices=PAYMENT_STATUS,
        default='PENDING'
    )
    transaction_id = models.CharField(max_length=255, unique=True)
    payment_date = models.DateTimeField(auto_now_add=True)
    last_modified = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.subscription.user.username} - {self.amount} {self.currency}"

    class Meta:
        db_table = 'subscription_payment'
