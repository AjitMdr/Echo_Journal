from rest_framework import serializers
from django.contrib.auth import get_user_model

User = get_user_model()  

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, style={'input_type': 'password'})
    password2 = serializers.CharField(write_only=True, required=True, style={'input_type': 'password'})

    class Meta:
        model = User
        fields = ['id', 'username', 'password', 'password2', 'email']

    def validate(self, data):
        if data['password'] != data['password2']:
            raise serializers.ValidationError({"password": "Passwords do not match."})
        return data

    def create(self, validated_data):
        validated_data.pop('password2')  # Remove password2 since it's not stored
        user = User.objects.create_user(
            email=validated_data['email'],
            username=validated_data['username'],
            password=validated_data['password']
        )
        return user


# User Profile Update Serializer (Includes Profile Picture)
class UserProfileUpdateSerializer(serializers.ModelSerializer):
    profile_picture = serializers.ImageField(required=False)  # Allow profile picture upload

    class Meta:
        model = User
        fields = ['username', 'email', 'profile_picture']  # Include profile_picture
        extra_kwargs = {
            'username': {'required': False},
            'email': {'required': False},
            'profile_picture': {'required': False},  # Optional profile picture
        }