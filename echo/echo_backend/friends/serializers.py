from rest_framework import serializers
from django.db import models
from .models import FriendRequest, Friendship
from django.contrib.auth import get_user_model
from django.db.models import Q

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']


class UserSearchSerializer(serializers.ModelSerializer):
    friendship_status = serializers.SerializerMethodField()
    profile_image = serializers.CharField(
        source='profile.image_url', read_only=True, default=None)
    last_seen = serializers.DateTimeField(
        source='profile.last_seen', read_only=True, default=None)

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name',
                  'friendship_status', 'profile_image', 'last_seen']

    def get_friendship_status(self, obj):
        # Return the friendship_status that was set in the viewset
        return getattr(obj, 'friendship_status', 'none')


class FriendRequestSerializer(serializers.ModelSerializer):
    to_user = UserSerializer(read_only=True)
    to_user_id = serializers.PrimaryKeyRelatedField(
        source='to_user',
        queryset=User.objects.all(),
        write_only=True
    )
    from_user = UserSerializer(read_only=True)
    status = serializers.CharField(read_only=True)
    created_at = serializers.DateTimeField(read_only=True)

    class Meta:
        model = FriendRequest
        fields = ['id', 'from_user', 'to_user', 'to_user_id', 'status', 'created_at']
        
    def validate(self, data):
        """
        Check that:
        1. User is not trying to send request to themselves
        2. Users are not already friends
        3. There isn't already a pending request between users
        """
        to_user = data['to_user']
        from_user = self.context['request'].user

        if to_user == from_user:
            raise serializers.ValidationError(
                "You cannot send a friend request to yourself.")

        # Check if users are already friends
        existing_friendship = Friendship.objects.filter(
            (Q(user1=from_user) & Q(user2=to_user)) |
            (Q(user1=to_user) & Q(user2=from_user))
        ).exists()
        if existing_friendship:
            raise serializers.ValidationError(
                "You are already friends with this user.")

        # Check for existing pending requests
        existing_request = FriendRequest.objects.filter(
            (Q(from_user=from_user) & Q(to_user=to_user)) |
            (Q(from_user=to_user) & Q(to_user=from_user)),
            status='pending'
        ).exists()
        if existing_request:
            raise serializers.ValidationError(
                "A friend request already exists between you and this user.")

        return data

    def create(self, validated_data):
        validated_data['from_user'] = self.context['request'].user
        validated_data['status'] = 'pending'
        return super().create(validated_data)


class FriendshipSerializer(serializers.ModelSerializer):
    user1 = UserSerializer(read_only=True)
    user2 = UserSerializer(read_only=True)

    class Meta:
        model = Friendship
        fields = ['id', 'user1', 'user2', 'created_at']

    def validate(self, data):
        """
        Check that:
        1. user1 and user2 are not the same user
        """
        user1 = self.context['request'].user
        user2 = data.get('user2', None)

        if user1 == user2:
            raise serializers.ValidationError(
                "Cannot create friendship with yourself.")

        return data
