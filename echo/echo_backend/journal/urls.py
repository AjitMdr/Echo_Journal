from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create a router and register our viewset
router = DefaultRouter()
router.register('', views.JournalViewSet, basename='journal')

# The API URLs are now determined automatically by the router
urlpatterns = router.urls
