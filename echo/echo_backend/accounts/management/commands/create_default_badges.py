from django.core.management.base import BaseCommand
from accounts.models import Badge


class Command(BaseCommand):
    help = 'Create default badges for the streak system'

    def handle(self, *args, **options):
        default_badges = [
            {
                'name': 'Beginner',
                'description': 'Complete 7 days of journaling',
                'badge_type': 'STREAK',
                'icon': '🔥',
                'requirement': 7
            },
            {
                'name': 'Consistent',
                'description': 'Complete 30 days of journaling',
                'badge_type': 'STREAK',
                'icon': '⚡',
                'requirement': 30
            },
            {
                'name': 'Dedicated',
                'description': 'Complete 100 days of journaling',
                'badge_type': 'STREAK',
                'icon': '🌟',
                'requirement': 100
            },
            {
                'name': 'Master',
                'description': 'Complete 365 days of journaling',
                'badge_type': 'STREAK',
                'icon': '👑',
                'requirement': 365
            }
        ]

        for badge_data in default_badges:
            badge, created = Badge.objects.get_or_create(
                name=badge_data['name'],
                defaults=badge_data
            )
            if created:
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Created badge: {badge_data["name"]}'
                    )
                )
            else:
                self.stdout.write(
                    self.style.WARNING(
                        f'Badge already exists: {badge_data["name"]}'
                    )
                )

        self.stdout.write(self.style.SUCCESS(
            'Default badges creation completed'))
