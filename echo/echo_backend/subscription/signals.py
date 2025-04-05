from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth import get_user_model
from django.utils import timezone
from .models import Subscription, Payment, Plan

User = get_user_model()


@receiver(post_save, sender=User)
def create_free_subscription(sender, instance, created, **kwargs):
    """Create a free subscription for new users"""
    if created:
        try:
            # Get or create the free plan
            free_plan, _ = Plan.objects.get_or_create(
                plan_type='FREE',
                defaults={
                    'name': 'Free Plan',
                    'price': 0,
                    'duration_days': 365,  # 1 year
                    'description': 'Basic features for free users',
                    'features': {
                        'basic_features': True,
                        'premium_features': False
                    }
                }
            )

            # Create subscription for the new user
            start_date = timezone.now()
            end_date = start_date + \
                timezone.timedelta(days=free_plan.duration_days)

            Subscription.objects.create(
                user=instance,
                plan=free_plan,
                status='ACTIVE',
                start_date=start_date,
                end_date=end_date,
                is_auto_renewal=True
            )
        except Exception as e:
            print(f"Error creating free subscription: {e}")


@receiver(post_save, sender=Payment)
def update_subscription_on_payment(sender, instance, created, **kwargs):
    """Update subscription status when payment is successful"""
    if created and instance.status == 'SUCCESS':
        subscription = instance.subscription
        if subscription.status != 'ACTIVE':
            subscription.status = 'ACTIVE'
            subscription.save()
