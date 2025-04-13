from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Streak, Badge, UserBadge

User = get_user_model()


class UserLoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        """
        Custom validation to check user authentication.
        """
        username = data.get('username')
        password = data.get('password')

        # Get user by email
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            raise serializers.ValidationError(
                "User with this username does not exist.")

        # Authenticate user
        if not user.check_password(password):
            raise serializers.ValidationError("Incorrect password.")

        if not user.is_active:
            raise serializers.ValidationError("This account is inactive.")

        # Check if account is verified
        if not user.is_verified:
            raise serializers.ValidationError(
                "Please verify your account first. Check your email for the verification OTP.")

        data['user'] = user  # Add user instance to validated data
        return data


class UserProfileUpdateSerializer(serializers.ModelSerializer):
    profile_picture = serializers.ImageField(
        required=False)  # Allow profile picture upload

    class Meta:
        model = User
        # Include profile_picture
        fields = ['username', 'email', 'profile_picture']
        extra_kwargs = {
            'username': {'required': False},
            'email': {'required': False},
            'profile_picture': {'required': False},  # Optional profile picture
        }


class UserSerializer(serializers.ModelSerializer):
    """Serializer for the User model"""
    class Meta:
        model = User
        fields = [
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'phone_number',
            'profile_picture',
            'is_verified',
            'is_active',
            'date_joined',
            'role'
        ]
        # Fields that shouldn't be modified through the API
        read_only_fields = ['id', 'is_active', 'date_joined', 'role']
        extra_kwargs = {
            # Password should never be readable
            'password': {'write_only': True}
        }

    def create(self, validated_data):
        """Create and return a new user"""
        password = validated_data.pop('password', None)
        user = super().create(validated_data)
        if password:
            user.set_password(password)
            user.save()
        return user

    def update(self, instance, validated_data):
        """Update and return an existing user"""
        password = validated_data.pop('password', None)
        user = super().update(instance, validated_data)
        if password:
            user.set_password(password)
            user.save()
        return user


class StreakSerializer(serializers.ModelSerializer):
    class Meta:
        model = Streak
        fields = ['current_streak', 'longest_streak', 'last_journal_date']


class BadgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Badge
        fields = ['id', 'name', 'description',
                  'badge_type', 'icon', 'requirement']


class UserBadgeSerializer(serializers.ModelSerializer):
    badge = BadgeSerializer(read_only=True)

    class Meta:
        model = UserBadge
        fields = ['badge', 'earned_at']
