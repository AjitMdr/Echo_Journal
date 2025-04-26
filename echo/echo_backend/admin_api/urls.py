from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    AdminDashboardViewSet,
    AdminLogViewSet,
    UserManagementViewSet
)

# Create a router for model-based viewsets
router = DefaultRouter()
router.register(r'logs', AdminLogViewSet, basename='logs')
router.register(r'users', UserManagementViewSet, basename='users')

# Dashboard endpoints
dashboard_urls = [
    path('metrics/',
         AdminDashboardViewSet.as_view({'get': 'metrics'}), name='dashboard-metrics'),
    path('trends/',
         AdminDashboardViewSet.as_view({'get': 'trends'}), name='dashboard-trends'),
    path('subscription_analytics/', AdminDashboardViewSet.as_view(
        {'get': 'subscription_analytics'}), name='subscription-analytics'),
    path('user_analytics/', AdminDashboardViewSet.as_view(
        {'get': 'user_analytics'}), name='user-analytics'),
    path('user_management/', AdminDashboardViewSet.as_view(
        {'get': 'user_management'}), name='user-management'),
    path('subscription_stats/', AdminDashboardViewSet.as_view(
        {'get': 'subscription_stats'}), name='subscription-stats'),
    path('transactions/', AdminDashboardViewSet.as_view(
        {'get': 'transactions'}), name='transactions'),
]

urlpatterns = [
    path('dashboard/', include(dashboard_urls)),
    path('', include(router.urls)),
]
