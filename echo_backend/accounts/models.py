from django.contrib.auth.models import AbstractUser
from django.db import models
from django.core.validators import EmailValidator

class User(AbstractUser):
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
    phone_number = models.CharField(max_length=15, blank=True, null=True)

    REQUIRED_FIELDS = ['username']
    USERNAME_FIELD = 'email'

    def __str__(self):
        return self.username
