import pytest
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import timedelta
import json
from decimal import Decimal

from admin_api.models import AdminLog, DashboardMetric
from journal.models import Journal
from subscription.models import Subscription, Plan, Payment

User = get_user_model()

@pytest.fixture
def api_client():
    return APIClient()

@pytest.fixture
def admin_user():
    user = User.objects.create_user(
        username='admin',
        email='admin@example.com',
        password='password123',
        is_staff=True
    )
    return user

@pytest.fixture
def regular_user():
    user = User.objects.create_user(
        username='regular',
        email='regular@example.com',
        password='password123'
    )
    return user

@pytest.fixture
def premium_plan():
    return Plan.objects.create(
        name='Premium Plan',
        plan_type='PREMIUM',
        price=Decimal('9.99'),
        description='Premium features'
    )

@pytest.fixture
def free_plan():
    return Plan.objects.create(
        name='Free Plan',
        plan_type='FREE',
        price=Decimal('0.00'),
        description='Basic features'
    )

@pytest.fixture
def active_subscription(regular_user, premium_plan):
    end_date = timezone.now() + timedelta(days=30)
    return Subscription.objects.create(
        user=regular_user,
        plan=premium_plan,
        status='ACTIVE',
        end_date=end_date,
        duration_days=30  # Adding the missing field
    )

@pytest.fixture
def payment(active_subscription):
    return Payment.objects.create(
        subscription=active_subscription,
        amount=active_subscription.plan.price,
        status='COMPLETED',
        payment_date=timezone.now()
    )

@pytest.fixture
def journal_entries(regular_user):
    journals = []
    # Create a few journal entries with different dates
    today = timezone.now()
    for i in range(5):
        entry_date = today - timedelta(days=i)
        journal = Journal.objects.create(
            user=regular_user,
            content=f"Journal entry {i}",
            date=entry_date
        )
        journals.append(journal)
    return journals

# Mock serializer for UserManagementViewSet
@pytest.fixture
def mock_user_serializer(monkeypatch):
    from admin_api.serializers import AdminUserSerializer
    from admin_api.views import UserManagementViewSet
    monkeypatch.setattr(UserManagementViewSet, 'serializer_class', AdminUserSerializer)

@pytest.mark.django_db
class TestAdminDashboardViewSet:

    # This test is done to verify unauthorized users can't access dashboard metrics
    def test_metrics_unauthorized(self, api_client):
        url = reverse('dashboard-metrics')
        response = api_client.get(url)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    # This test is done to ensure that non-admin users are forbidden from accessing dashboard metrics
    def test_metrics_not_admin(self, api_client, regular_user):
        url = reverse('dashboard-metrics')
        api_client.force_authenticate(user=regular_user)
        response = api_client.get(url)
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    # This test is done to verify that admin users can access trend data like journal activity and revenue trends
    def test_trends(self, api_client, admin_user, journal_entries):
        url = reverse('dashboard-trends')
        api_client.force_authenticate(user=admin_user)
        response = api_client.get(url)
        
        assert response.status_code == status.HTTP_200_OK
        assert 'journal_activity' in response.data
        assert 'user_growth' in response.data
        assert 'revenue_trend' in response.data
        assert 'engagement_trend' in response.data
    
    # This test is done to verify the user analytics data for an admin, including activity and retention metrics
    def test_user_analytics(self, api_client, admin_user, regular_user, journal_entries):
        url = reverse('user-analytics')
        api_client.force_authenticate(user=admin_user)
        response = api_client.get(url)
        
        assert response.status_code == status.HTTP_200_OK
        assert 'activity_by_hour' in response.data
        assert 'engagement_by_day' in response.data
        assert 'retention_metrics' in response.data
        
        # Verify user count
        total_users = response.data['retention_metrics']['total_users']
        assert total_users == 2  # admin + regular
    
    # This test is done to check that user list is returned correctly for an admin in the user management view
    def test_user_management(self, api_client, admin_user, regular_user):
        url = reverse('user-management')
        api_client.force_authenticate(user=admin_user)
        response = api_client.get(url)
        
        assert response.status_code == status.HTTP_200_OK
        assert 'users' in response.data
        assert len(response.data['users']) == 2  # admin + regular

    

   

@pytest.mark.django_db
class TestUserManagementViewSet:
    
    # This test is done to verify unauthorized users cannot access user list
    def test_user_list_unauthorized(self, api_client):
        url = reverse('users-list')
        response = api_client.get(url)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    # This test is done to verify that non-admin users are not allowed to access the user list
    def test_user_list_not_admin(self, api_client, regular_user):
        url = reverse('users-list')
        api_client.force_authenticate(user=regular_user)
        response = api_client.get(url)
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    # This test is done to verify that admin users can successfully retrieve the user list
    def test_user_list_admin(self, api_client, admin_user, regular_user, mock_user_serializer):
        url = reverse('users-list')
        api_client.force_authenticate(user=admin_user)
        response = api_client.get(url)
        
        assert response.status_code == status.HTTP_200_OK
     
   