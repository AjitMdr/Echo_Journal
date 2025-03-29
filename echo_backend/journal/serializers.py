from rest_framework import serializers
from .models import Journal
from django.contrib.auth import get_user_model


class JournalSerializer(serializers.ModelSerializer):
    """Serializer for retrieving journal entries"""
    username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = Journal
        fields = [
            'id',
            'title',
            'content',
            'language',
            'date',
            'edit_date',
            'username'
        ]
        read_only_fields = ['id', 'date', 'edit_date', 'username']


class JournalCreateUpdateSerializer(serializers.ModelSerializer):
    """Serializer for creating and updating journal entries"""

    class Meta:
        model = Journal
        fields = ['title', 'content', 'language']

    def validate_title(self, value):
        """Validate the title field"""
        if len(value.strip()) < 3:
            raise serializers.ValidationError(
                "Title must be at least 3 characters long")
        if len(value) > 255:
            raise serializers.ValidationError(
                "Title must not exceed 255 characters")
        return value.strip()

    def validate_content(self, value):
        """Validate the content field"""
        if len(value.strip()) < 10:
            raise serializers.ValidationError(
                "Content must be at least 10 characters long")
        return value.strip()

    def validate_language(self, value):
        """Validate the language field"""
        valid_languages = ['en', 'ne', 'es',
                           'fr', 'de']  # Added 'ne' for Nepali
        if value not in valid_languages:
            raise serializers.ValidationError(
                f"Language must be one of: {', '.join(valid_languages)}")
        return value
