from django.db import models
from django.contrib.auth.models import User

class Journal(models.Model):
    user = models.ForeignKey(User, related_name='journals', on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    content = models.TextField()
    language = models.CharField(max_length=20, default='en')
    date = models.DateTimeField(auto_now_add=True)
    edit_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.title} - {self.user.username}'
