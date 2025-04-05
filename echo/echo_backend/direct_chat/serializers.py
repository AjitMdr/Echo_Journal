from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import DirectMessage, Conversation

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']


class DirectMessageSerializer(serializers.ModelSerializer):
    sender_username = serializers.SerializerMethodField()
    receiver_username = serializers.SerializerMethodField()
    conversation_id = serializers.SerializerMethodField()
    message = serializers.SerializerMethodField()
    type = serializers.SerializerMethodField()

    class Meta:
        model = DirectMessage
        fields = ['id', 'sender', 'sender_username', 'receiver', 'receiver_username',
                  'content', 'message', 'timestamp', 'is_read', 'conversation_id', 'type']
        read_only_fields = ['sender', 'timestamp']

    def get_sender_username(self, obj):
        return obj.sender.username

    def get_receiver_username(self, obj):
        return obj.receiver.username

    def get_conversation_id(self, obj):
        if obj.conversation:
            return obj.conversation.id
        return None

    def get_message(self, obj):
        return obj.content

    def get_type(self, obj):
        return 'chat_message'


class ConversationSerializer(serializers.ModelSerializer):
    participants = UserSerializer(many=True, read_only=True)
    last_message = serializers.SerializerMethodField()
    unread_count = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = ['id', 'participants', 'created_at',
                  'updated_at', 'last_message', 'unread_count']

    def get_last_message(self, obj):
        last_message = obj.get_last_message()
        if last_message:
            return DirectMessageSerializer(last_message).data
        return None

    def get_unread_count(self, obj):
        user = self.context.get('request').user
        return obj.get_unread_count(user)
