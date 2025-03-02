from rest_framework import serializers
from .models import Journal

class JournalSerializer(serializers.ModelSerializer):
    """Serializer for retrieving journal entries"""

    class Meta:
        model = Journal
        fields = ['id', 'user', 'title', 'content', 'language', 'date', 'edit_date']
        read_only_fields = ['id', 'user', 'date', 'edit_date']


class JournalCreateUpdateSerializer(serializers.ModelSerializer):
    """Serializer for creating and updating journal entries"""

    class Meta:
        model = Journal
        fields = ['title', 'content', 'language']
