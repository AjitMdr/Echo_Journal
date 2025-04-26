from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import FriendRequestViewSet, FriendshipViewSet, UserSearchViewSet

app_name = 'friends'  # Add this line to create the namespace

router = DefaultRouter()
router.register(r'requests', FriendRequestViewSet, basename='friendrequest')  # Change basename to match test
router.register(r'friendships', FriendshipViewSet, basename='friendship')
router.register(r'search', UserSearchViewSet, basename='usersearch')  # Consider removing the dash here too

urlpatterns = [
    path('', include(router.urls)),
]