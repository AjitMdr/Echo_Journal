# Fixes for test_views.py

import pytest
import json
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal
from subscription.models import Plan, Subscription, Payment

User = get_user_model()

@pytest.fixture
def api_client():
    return APIClient()

@pytest.fixture
def user():
    return User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='password123'
    )

@pytest.fixture
def authenticated_client(api_client, user):
    api_client.force_authenticate(user=user)
    return api_client

@pytest.fixture
def plan():
    return Plan.objects.create(
        name='Test Premium',
        plan_type='PREMIUM',
        price=Decimal('99.99'),
        duration_days=30,
        description='Test premium plan',
        features={'feature1': True, 'feature2': True},
        is_active=True
    )

@pytest.fixture
def inactive_plan():
    return Plan.objects.create(
        name='Inactive Plan',
        plan_type='PREMIUM',
        price=Decimal('79.99'),
        duration_days=30,
        description='Inactive plan',
        features={'feature1': True},
        is_active=False
    )

@pytest.fixture
def subscription(user, plan):
    return Subscription.objects.create(
        user=user,
        plan=plan,
        status='ACTIVE',
        start_date=timezone.now(),
        end_date=timezone.now() + timedelta(days=30),
        is_auto_renewal=False
    )

@pytest.fixture
def payment(subscription):
    return Payment.objects.create(
        subscription=subscription,
        amount=subscription.plan.price,
        currency='NPR',
        payment_method='ESEWA',
        status='SUCCESS',
        transaction_id='test-transaction-123'
    )

# Test class for Plan ViewSet
@pytest.mark.django_db
class TestPlanViewSet:
    # Test scenario: Authenticated users can list active plans
    def test_list_plans_authenticated(self, authenticated_client, plan, inactive_plan):
        url = reverse('subscription:plan-list')
        response = authenticated_client.get(url)
        
        assert response.status_code == status.HTTP_200_OK
        # Find our specific plan in the response data
        plan_data = next((item for item in response.data if item['id'] == plan.id), None)
        assert plan_data is not None
        assert plan_data['name'] == plan.name
        
    # Test scenario: Unauthenticated users cannot list plans
    def test_list_plans_unauthenticated(self, api_client, plan):
        url = reverse('subscription:plan-list')
        response = api_client.get(url)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        

# Test class for Subscription ViewSet
@pytest.mark.django_db
class TestSubscriptionViewSet:
    
        
    # Test scenario: Authenticated users can create a new subscription
    def test_create_subscription(self, authenticated_client, plan):
        url = reverse('subscription:subscription-list')
        data = {
            'plan': plan.id,
            'is_auto_renewal': True
        }
        response = authenticated_client.post(url, data, format='json')
        
        assert response.status_code == status.HTTP_201_CREATED
        assert response.data['plan'] == plan.id
        assert response.data['status'] == 'ACTIVE'
        assert response.data['is_auto_renewal'] == True
        
    # Test scenario: Authenticated users can renew a subscription
    def test_renew_subscription(self, authenticated_client, subscription):
        url = reverse('subscription:subscription-renew', kwargs={'pk': subscription.id})
        original_end_date = subscription.end_date
        
        # Set auto renewal to true for test
        subscription.is_auto_renewal = True
        subscription.save()
        
        response = authenticated_client.post(url)
        
        assert response.status_code == status.HTTP_200_OK
        
        # Check that subscription dates were updated
        subscription.refresh_from_db()
        assert subscription.start_date == original_end_date
        assert subscription.end_date > original_end_date

# Test class for Payment ViewSet
@pytest.mark.django_db
class TestPaymentViewSet:
    
    # Test scenario: Authenticated users can create a payment for a subscription
    def test_create_payment(self, authenticated_client, plan):
        url = reverse('subscription:payment-list')
        data = {
            'plan': plan.id,
            'transaction_id': 'new-transaction-123',
            'payment_method': 'ESEWA',
            'amount': plan.price,
            'currency': 'NPR'
        }
        response = authenticated_client.post(url, data, format='json')
        
        assert response.status_code == status.HTTP_201_CREATED
        
        # Check that both payment and subscription were created
        payment_id = response.data['id']
        payment = Payment.objects.get(id=payment_id)
        assert payment.transaction_id == 'new-transaction-123'
        assert payment.amount == plan.price
        
        assert payment.subscription is not None
        assert payment.subscription.user.username == 'testuser'
        assert payment.subscription.plan.id == plan.id
        assert payment.subscription.status == 'ACTIVE'
        
    # Test scenario: Payment creation fails if plan is missing
    def test_create_payment_missing_plan(self, authenticated_client):
        url = reverse('subscription:payment-list')
        data = {
            'transaction_id': 'missing-plan-transaction',
            'payment_method': 'ESEWA',
            'amount': Decimal('99.99'),
            'currency': 'NPR'
        }
        response = authenticated_client.post(url, data, format='json')
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
    # Test scenario: Payment creation fails if a transaction ID is duplicated
    def test_create_payment_duplicate_transaction(self, authenticated_client, payment, plan):
        url = reverse('subscription:payment-list')
        data = {
            'plan': plan.id,
            'transaction_id': payment.transaction_id,  # Using existing transaction_id
            'payment_method': 'ESEWA',
            'amount': plan.price,
            'currency': 'NPR'
        }
        response = authenticated_client.post(url, data, format='json')
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        error_message = str(response.data)
        assert 'payment with this transaction id already exists' in error_message 