from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AdminDashboardViewSet, AdminLogViewSet, UserManagementViewSet

router = DefaultRouter()
router.register(r'dashboard', AdminDashboardViewSet,
                basename='admin-dashboard')
router.register(r'logs', AdminLogViewSet)
router.register(r'users', UserManagementViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
