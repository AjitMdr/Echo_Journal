from django.shortcuts import render, get_object_or_404
from django.contrib.auth import get_user_model
from django.db.models import Q
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import DirectMessage, Conversation
from .serializers import DirectMessageSerializer, ConversationSerializer, UserSerializer
import re
from django.http import JsonResponse
from .routing import websocket_urlpatterns

User = get_user_model()


class ConversationViewSet(viewsets.ModelViewSet):
    serializer_class = ConversationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(participants=self.request.user)

    def create(self, request):
        # Get or create a conversation between the current user and the specified user
        user_id = request.data.get('user_id')
        if not user_id:
            return Response({"error": "user_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        other_user = get_object_or_404(User, id=user_id)

        # Check if a conversation already exists
        conversations = Conversation.objects.filter(
            participants=request.user).filter(participants=other_user)

        if conversations.exists():
            serializer = self.get_serializer(conversations.first())
            return Response(serializer.data)

        # Create a new conversation
        conversation = Conversation.objects.create()
        conversation.participants.add(request.user, other_user)

        serializer = self.get_serializer(conversation)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        conversation = self.get_object()
        messages = conversation.get_messages()

        # Mark messages as read
        conversation.mark_messages_as_read(request.user)

        serializer = DirectMessageSerializer(messages, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def send_message(self, request, pk=None):
        conversation = self.get_object()
        content = request.data.get('content')

        if not content:
            return Response({"error": "content is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Get the other participant
        other_user = conversation.participants.exclude(
            id=request.user.id).first()

        if not other_user:
            return Response({"error": "Conversation must have another participant"}, status=status.HTTP_400_BAD_REQUEST)

        # Create the message with the conversation relationship
        message = DirectMessage.objects.create(
            sender=request.user,
            receiver=other_user,
            content=content,
            conversation=conversation  # Link the message to the conversation
        )

        # Update the conversation's updated_at timestamp
        conversation.save()

        serializer = DirectMessageSerializer(message)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['get'])
    def recent(self, request):
        conversations = self.get_queryset().order_by('-updated_at')[:10]
        serializer = self.get_serializer(
            conversations, many=True, context={'request': request})
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def mark_read(self, request, pk=None):
        """Mark all messages in the conversation as read for the current user"""
        conversation = self.get_object()

        # Mark all messages as read
        conversation.mark_messages_as_read(request.user)

        return Response({"status": "success"}, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'])
    def unread_count(self, request):
        """Get the total count of unread messages across all conversations"""
        # Get all conversations for the current user
        conversations = self.get_queryset()

        # Count all unread messages
        total_unread = 0
        for conversation in conversations:
            # Get the other participant
            other_user = conversation.participants.exclude(
                id=request.user.id).first()
            if other_user:
                # Count unread messages from this user
                unread_count = DirectMessage.objects.filter(
                    conversation=conversation,
                    sender=other_user,
                    receiver=request.user,
                    is_read=False
                ).count()
                total_unread += unread_count

        return Response({"unread_count": total_unread})


class DirectMessageViewSet(viewsets.ModelViewSet):
    serializer_class = DirectMessageSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return DirectMessage.objects.filter(
            Q(sender=self.request.user) | Q(receiver=self.request.user)
        ).order_by('timestamp')

    def perform_create(self, serializer):
        serializer.save(sender=self.request.user)

    @action(detail=False, methods=['get'])
    def with_user(self, request):
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({"error": "user_id query parameter is required"}, status=status.HTTP_400_BAD_REQUEST)

        other_user = get_object_or_404(User, id=user_id)

        messages = DirectMessage.objects.filter(
            (Q(sender=request.user) & Q(receiver=other_user)) |
            (Q(sender=other_user) & Q(receiver=request.user))
        ).order_by('timestamp')

        # Mark messages as read
        DirectMessage.objects.filter(
            sender=other_user, receiver=request.user, is_read=False).update(is_read=True)

        serializer = self.get_serializer(messages, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def unread_count(self, request):
        count = DirectMessage.objects.filter(
            receiver=request.user, is_read=False).count()
        return Response({"unread_count": count})


def check_websocket_path(request, test_path):
    """
    Diagnostic endpoint to check if a WebSocket path would match our routing patterns
    """
    if not test_path.startswith('/'):
        test_path = '/' + test_path

    results = []
    for pattern in websocket_urlpatterns:
        pattern_str = pattern.pattern.pattern
        match = re.match(pattern_str, test_path)

        if match:
            groups = match.groups()
            pattern_result = {
                'pattern': pattern_str,
                'matched': True,
                'groups': groups,
                'kwargs': match.groupdict()
            }
        else:
            pattern_result = {
                'pattern': pattern_str,
                'matched': False
            }

        results.append(pattern_result)

    # Try to extract friend_id using the pattern directly
    friend_id_pattern = r'^ws/chat/direct/(?P<friend_id>\d+)/?$'
    friend_id_match = re.match(friend_id_pattern, test_path)
    friend_id = friend_id_match.groupdict(
    )['friend_id'] if friend_id_match else None

    return JsonResponse({
        'path': test_path,
        'path_parts': test_path.strip('/').split('/'),
        'would_match': any(result['matched'] for result in results),
        'results': results,
        'extracted_friend_id': friend_id
    })
