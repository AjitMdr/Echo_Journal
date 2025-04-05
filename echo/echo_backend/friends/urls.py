from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import FriendRequestViewSet, FriendshipViewSet, UserSearchViewSet

router = DefaultRouter()
router.register(r'requests', FriendRequestViewSet, basename='friend-request')
router.register(r'friendships', FriendshipViewSet, basename='friendship')
router.register(r'search', UserSearchViewSet, basename='user-search')

urlpatterns = [
    path('', include(router.urls)),
]
