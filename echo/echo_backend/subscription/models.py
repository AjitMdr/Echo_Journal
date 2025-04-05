from django.db import models
from django.conf import settings
from django.utils import timezone
from django.contrib.auth import get_user_model

User = get_user_model()


class Plan(models.Model):
    """Subscription plan model"""
    PLAN_TYPES = [
        ('FREE', 'Free'),
        ('PREMIUM', 'Premium'),
    ]

    name = models.CharField(max_length=100)
    plan_type = models.CharField(max_length=20, choices=PLAN_TYPES)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    duration_days = models.IntegerField()  # Duration in days
    description = models.TextField()
    features = models.JSONField(default=dict)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.plan_type})"

    class Meta:
        db_table = 'subscription_plan'


class Subscription(models.Model):
    """User subscription model"""
    STATUS_CHOICES = [
        ('ACTIVE', 'Active'),
        ('EXPIRED', 'Expired'),
        ('CANCELLED', 'Cancelled'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    plan = models.ForeignKey(Plan, on_delete=models.PROTECT)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default='ACTIVE')
    start_date = models.DateTimeField(auto_now_add=True)
    end_date = models.DateTimeField()
    is_auto_renewal = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def is_active(self):
        return self.status == 'ACTIVE'

    @property
    def days_remaining(self):
        if not self.is_active:
            return 0
        delta = self.end_date - timezone.now()
        return max(0, delta.days)

    def __str__(self):
        return f"{self.user.username}'s {self.plan.name} subscription"

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
        ('ESEWA', 'eSewa'),
    ]

    subscription = models.ForeignKey(
        Subscription,
        on_delete=models.PROTECT,
        related_name='payments',
        null=True,  # Allow null for initial payment creation
        blank=True
    )
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=3, default='NPR')
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
