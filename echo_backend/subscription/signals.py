from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Subscription, Payment


@receiver(post_save, sender=Payment)
def update_subscription_on_payment(sender, instance, created, **kwargs):
    """Update subscription status when payment is successful"""
    if created and instance.status == 'SUCCESS':
        subscription = instance.subscription
        if subscription.status != 'ACTIVE':
            subscription.status = 'ACTIVE'
            subscription.save() 