from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.utils import timezone
from subscription.models import Plan, Subscription

User = get_user_model()


class Command(BaseCommand):
    help = 'Creates subscription plans and assigns free subscriptions to users who do not have one'

    def handle(self, *args, **kwargs):
        # First, create or update the free plan
        free_plan, created = Plan.objects.get_or_create(
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

        # Create or update the premium plan
        premium_plan, premium_created = Plan.objects.get_or_create(
            plan_type='PREMIUM',
            defaults={
                'name': 'Premium Plan',
                'price': 9.99,  # Monthly subscription fee
                'duration_days': 30,  # Monthly plan
                'description': 'All features including AI-powered mood analysis',
                'features': {
                    'basic_features': True,
                    'premium_features': True
                }
            }
        )

        if created:
            self.stdout.write(self.style.SUCCESS('Created free plan'))
        else:
            self.stdout.write(self.style.SUCCESS('Using existing free plan'))

        if premium_created:
            self.stdout.write(self.style.SUCCESS('Created premium plan'))
        else:
            self.stdout.write(self.style.SUCCESS('Using existing premium plan'))

        # Get all users
        users = User.objects.all()
        created_count = 0
        existing_count = 0

        for user in users:
            # Check if user already has a subscription
            if not Subscription.objects.filter(user=user).exists():
                # Create free subscription for user
                start_date = timezone.now()
                end_date = start_date + \
                    timezone.timedelta(days=free_plan.duration_days)

                Subscription.objects.create(
                    user=user,
                    plan=free_plan,
                    status='ACTIVE',
                    start_date=start_date,
                    end_date=end_date,
                    is_auto_renewal=True
                )
                created_count += 1
            else:
                existing_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully processed {len(users)} users:\n'
                f'- Created {created_count} new free subscriptions\n'
                f'- Found {existing_count} existing subscriptions\n'
                f'\nAvailable Plans:\n'
                f'1. Free Plan: $0/year\n'
                f'2. Premium Plan: ${premium_plan.price}/month with mood analysis'
            )
        )
