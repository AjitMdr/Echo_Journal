from django.urls import re_path
from . import consumers

# WebSocket URL patterns for direct chat
# Use a single, consistent pattern for WebSocket connections
websocket_urlpatterns = [
    # Match only paths with pure digit-only friend_id
    re_path(r'^ws/chat/direct/(?P<friend_id>\d+)/?$', consumers.DirectChatConsumer.as_asgi()),
]
