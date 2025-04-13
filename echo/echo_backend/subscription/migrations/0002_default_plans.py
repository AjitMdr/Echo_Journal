from django.db import migrations


def create_default_plans(apps, schema_editor):
    Plan = apps.get_model('subscription', 'Plan')

    # First, delete any existing plans
    Plan.objects.all().delete()

    # Create Free Plan (Default)
    Plan.objects.create(
        name='Free Plan',
        plan_type='FREE',
        price= 0.00,
        duration_days=365,  # Free plan is valid for a year and auto-renews
        description='All features except mood analysis',
        features={
            'journal_entries': True,  # Basic journaling
            'friends': True,          # Connect with friends
            'chat': True,            # Chat with friends
            'mood_analysis': False,   # AI-powered mood analysis (premium only)
            'analytics': True         # Basic analytics
        },
        is_active=True
    )

    # Create Premium Plan
    Plan.objects.create(
        name='Premium Plan',
        plan_type='PREMIUM',
        price=990,
        duration_days=30,
        description='All features including AI-powered mood analysis',
        features={
            'journal_entries': True,  # Basic journaling
            'friends': True,          # Connect with friends
            'chat': True,            # Chat with friends
            'mood_analysis': True,    # AI-powered mood analysis
            'analytics': True         # Basic analytics
        },
        is_active=True
    )


def reverse_migration(apps, schema_editor):
    Plan = apps.get_model('subscription', 'Plan')
    Plan.objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [
        ('subscription', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(create_default_plans, reverse_migration),
    ]
