from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.db import transaction

User = get_user_model()


class Command(BaseCommand):
    help = 'Creates a superadmin user'

    def add_arguments(self, parser):
        parser.add_argument('--username', type=str, help='Superadmin username')
        parser.add_argument('--email', type=str, help='Superadmin email')
        parser.add_argument('--password', type=str, help='Superadmin password')

    def handle(self, *args, **options):
        username = options['username']
        email = options['email']
        password = options['password']

        if not all([username, email, password]):
            self.stdout.write(
                self.style.ERROR(
                    'All fields are required (username, email, password)')
            )
            return

        try:
            with transaction.atomic():
                # Check if superadmin exists
                if User.objects.filter(is_superuser=True).exists():
                    self.stdout.write(
                        self.style.WARNING('A superadmin user already exists')
                    )
                    return

                # Create superadmin
                user = User.objects.create_superuser(
                    username=username,
                    email=email,
                    password=password
                )

                # Set additional fields
                user.is_verified = True
                user.role = 'SUPERADMIN'
                user.save()

                self.stdout.write(
                    self.style.SUCCESS(
                        f'Successfully created superadmin user: {username}')
                )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error creating superadmin: {str(e)}')
            )
