from django.urls import path, include, re_path
from rest_framework.routers import DefaultRouter
from .views import ConversationViewSet, DirectMessageViewSet, check_websocket_path
from .routing import websocket_urlpatterns

router = DefaultRouter()
router.register(r'conversations', ConversationViewSet, basename='conversation')
router.register(r'messages', DirectMessageViewSet, basename='message')

urlpatterns = [
    path('', include(router.urls)),
    path('check-websocket/<path:test_path>/',
         check_websocket_path, name='check-websocket'),
]
