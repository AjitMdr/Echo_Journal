# tests/test_auth_api.py
import pytest
import json
from django.urls import reverse
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status
from django.core.cache import cache
from django.utils import timezone
from datetime import timedelta
from unittest.mock import patch, MagicMock

User = get_user_model()

@pytest.fixture
def api_client():
    return APIClient()

@pytest.fixture
def create_user():
    def _create_user(username="testuser", email="test@example.com", password="testpassword123", is_verified=True):
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            is_verified=is_verified
        )
        return user
    return _create_user

@pytest.fixture
def auth_user(create_user, api_client):
    user = create_user()
    url = reverse('login')
    response = api_client.post(url, {'username': user.username, 'password': 'testpassword123'})
    token = response.data['access']
    api_client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
    return user

@pytest.mark.django_db
class TestRegistration:
    def test_register_success(self, api_client, monkeypatch):
        # This test is done for successful user registration
        print("Test: User registration with valid data should succeed")
        monkeypatch.setattr('accounts.views.send_otp_email', lambda email, otp: True)
        url = reverse('register')
        data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'newpassword123'
        }
        response = api_client.post(url, data, format='json')
        assert response.status_code == status.HTTP_201_CREATED
        assert 'message' in response.data
        assert 'email' in response.data
        assert User.objects.filter(username='newuser').exists()

    def test_register_duplicate_username(self, api_client, create_user, monkeypatch):
        # This test is done for user registration with duplicate username
        print("Test: User registration with duplicate username should fail")
        create_user(username='existinguser')
        monkeypatch.setattr('accounts.views.send_otp_email', lambda email, otp: True)
        url = reverse('register')
        data = {
            'username': 'existinguser',
            'email': 'different@example.com',
            'password': 'newpassword123'
        }
        response = api_client.post(url, data, format='json')
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert 'error' in response.data

    def test_register_duplicate_email(self, api_client, create_user, monkeypatch):
        # This test is done for user registration with duplicate email
        print("Test: User registration with duplicate email should fail")
        create_user(email='existing@example.com')
        monkeypatch.setattr('accounts.views.send_otp_email', lambda email, otp: True)
        url = reverse('register')
        data = {
            'username': 'newuser',
            'email': 'existing@example.com',
            'password': 'newpassword123'
        }
        response = api_client.post(url, data, format='json')
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert 'error' in response.data

    def test_register_missing_fields(self, api_client):
        # This test is done for user registration with missing fields
        print("Test: User registration with missing fields should fail")
        url = reverse('register')
        data = {
            'username': 'newuser',
        }
        response = api_client.post(url, data, format='json')
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert 'error' in response.data

@pytest.mark.django_db
class TestEmailVerification:
    def test_verify_otp_success(self, api_client, create_user, monkeypatch):
        # This test is done for successful email verification via OTP
        print("Test: Email verification with valid OTP should succeed")
        user = create_user(is_verified=False)
        def mock_cache_get(key):
            if key == f"verify_email_otp_{user.email}":
                return {'otp': '123456', 'user_id': user.id}
            return None
        monkeypatch.setattr(cache, 'get', mock_cache_get)
        monkeypatch.setattr(cache, 'delete', lambda key: None)
        url = reverse('verify-otp-and-signup')
        data = {
            'email': user.email,
            'otp': '123456'
        }
        response = api_client.post(url, data, format='json')
        assert response.status_code == status.HTTP_200_OK
        assert 'token' in response.data
        user.refresh_from_db()
        assert user.is_verified is True

    def test_verify_otp_invalid(self, api_client, create_user, monkeypatch):
        # This test is done for email verification with invalid OTP
        print("Test: Email verification with invalid OTP should fail")
        user = create_user(is_verified=False)
        def mock_cache_get(key):
            if key == f"verify_email_otp_{user.email}":
                return {'otp': '123456', 'user_id': user.id}
            return None
        monkeypatch.setattr(cache, 'get', mock_cache_get)
        url = reverse('verify-otp-and-signup')
        data = {
            'email': user.email,
            'otp': '654321'
        }
        response = api_client.post(url, data, format='json')
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert 'error' in response.data
        user.refresh_from_db()
        assert user.is_verified is False

    def test_verify_otp_expired(self, api_client, create_user, monkeypatch):
        # This test is done for email verification with expired OTP
        print("Test: Email verification with expired OTP should fail")
        user = create_user(is_verified=False)
        monkeypatch.setattr(cache, 'get', lambda key: None)
        url = reverse('verify-otp-and-signup')
        data = {
            'email': user.email,
            'otp': '123456'
        }
        response = api_client.post(url, data, format='json')
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert 'error' in response.data
        user.refresh_from_db()
        assert user.is_verified is False

@pytest.mark.django_db
class TestLogin:
    def test_login_success(self, api_client, create_user):
        # This test is done for successful user login
        print("Test: Login with valid credentials should succeed")
        user = create_user()
        url = reverse('login')
        data = {
            'username': user.username,
            'password': 'testpassword123'
        }
        response = api_client.post(url, data, format='json')
        assert response.status_code == status.HTTP_200_OK
        assert 'access' in response.data
        assert 'refresh' in response.data
        assert 'user' in response.data

    def test_login_unverified_user(self, api_client, create_user):
        # This test is done for login attempt with unverified user
        print("Test: Login with unverified account should fail")
        user = create_user(is_verified=False)
        url = reverse('login')
        data = {
            'username': user.username,
            'password': 'testpassword123'
        }
        response = api_client.post(url, data, format='json')
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert 'error' in response.data
        assert 'needs_verification' in response.data

    def test_login_incorrect_password(self, api_client, create_user):
        # This test is done for login attempt with incorrect password
        print("Test: Login with incorrect password should fail")
        user = create_user()
        url = reverse('login')
        data = {
            'username': user.username,
            'password': 'wrongpassword'
        }
        response = api_client.post(url, data, format='json')
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert 'error' in response.data

@pytest.mark.django_db
class TestTwoFactorAuth:
    def test_2fa_flow(self, api_client, create_user, monkeypatch):
        # This test is done for full two-factor authentication flow
        print("Test: Two-factor authentication flow should work correctly")
        user = create_user()
        user.two_factor_enabled = True
        user.save()
        monkeypatch.setattr('accounts.views.send_2fa_login_otp', lambda email, otp: True)
        def mock_cache_set(key, value, timeout): return True
        def mock_cache_get(key):
            if key == f"2fa_login_otp_{user.email}":
                return {'otp': '123456', 'user_id': user.id}
            return None
        monkeypatch.setattr(cache, 'set', mock_cache_set)
        monkeypatch.setattr(cache, 'get', mock_cache_get)
        monkeypatch.setattr(cache, 'delete', lambda key: None)

        login_url = reverse('login')
        login_data = {
            'username': user.username,
            'password': 'testpassword123'
        }
        login_response = api_client.post(login_url, login_data, format='json')
        assert login_response.status_code == status.HTTP_200_OK
        assert 'requires_2fa' in login_response.data
        assert login_response.data['requires_2fa'] is True

        verify_url = reverse('verify-2fa-login')
        verify_data = {
            'email': user.email,
            'otp': '123456'
        }
        verify_response = api_client.post(verify_url, verify_data, format='json')
        assert verify_response.status_code == status.HTTP_200_OK
        assert 'access' in verify_response.data
        assert 'refresh' in verify_response.data

    def test_toggle_2fa(self, api_client, auth_user):
        # This test is done for toggling 2FA setting on and off
        print("Test: Toggling 2FA setting should work correctly")
        url = reverse('two-factor-toggle')
        enable_data = {'enable': True}
        enable_response = api_client.post(url, enable_data, format='json')
        assert enable_response.status_code == status.HTTP_200_OK
        assert enable_response.data['is_enabled'] is True
        auth_user.refresh_from_db()
        assert auth_user.two_factor_enabled is True

        disable_data = {'enable': False}
        disable_response = api_client.post(url, disable_data, format='json')
        assert disable_response.status_code == status.HTTP_200_OK
        assert disable_response.data['is_enabled'] is False
        auth_user.refresh_from_db()
        assert auth_user.two_factor_enabled is False

@pytest.mark.django_db
class TestPasswordReset:
    def test_forgot_password_flow(self, api_client, create_user, monkeypatch):
        # This test is done for full forgot-password and reset flow
        print("Test: Password reset flow should work correctly")
        user = create_user()
        monkeypatch.setattr('accounts.views.send_password_reset_otp_email', lambda email, otp: True)
        def mock_cache_set(key, value, timeout): return True
        def mock_cache_get(key):
            if key == f"password_reset_otp_{user.email}":
                return {'otp': '123456', 'user_id': user.id}
            return None
        monkeypatch.setattr(cache, 'set', mock_cache_set)
        monkeypatch.setattr(cache, 'get', mock_cache_get)
        monkeypatch.setattr(cache, 'delete', lambda key: None)

        forgot_url = reverse('forgot-password')
        forgot_data = {'email': user.email}
        forgot_response = api_client.post(forgot_url, forgot_data, format='json')
        assert forgot_response.status_code == status.HTTP_200_OK
        assert forgot_response.data['success'] is True

        reset_url = reverse('verify-otp-reset-password')
        reset_data = {
            'email': user.email,
            'otp': '123456',
            'new_password': 'newpassword123'
        }
        reset_response = api_client.post(reset_url, reset_data, format='json')
        assert reset_response.status_code == status.HTTP_200_OK
        assert reset_response.data['success'] is True
        assert 'access' in reset_response.data

        login_url = reverse('login')
        login_data = {
            'username': user.username,
            'password': 'newpassword123'
        }
        login_response = api_client.post(login_url, login_data, format='json')
        assert login_response.status_code == status.HTTP_200_OK

@pytest.mark.django_db
class TestUserProfile:
    def test_get_profile(self, api_client, auth_user):
        # This test is done for retrieving the authenticated user's profile
        print("Test: Getting user profile should work correctly")
        url = reverse('get_profile')
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert 'id' in response.data
        assert 'username' in response.data
        assert 'email' in response.data
        assert response.data['id'] == auth_user.id
        assert response.data['username'] == auth_user.username

    def test_update_profile(self, api_client, auth_user):
        # This test is done for updating the authenticated user's profile
        print("Test: Updating user profile should work correctly")
        url = reverse('update_profile')
        data = {
            'username': 'updatedusername',
            'email': 'updated@example.com'
        }
        response = api_client.put(url, data, format='json')
        assert response.status_code == status.HTTP_200_OK
        assert response.data['username'] == 'updatedusername'
        assert response.data['email'] == 'updated@example.com'
        auth_user.refresh_from_db()
        assert auth_user.username == 'updatedusername'
        assert auth_user.email == 'updated@example.com'

    def test_change_password(self, api_client, auth_user):
        # This test is done for changing the user's password
        print("Test: Changing password should work correctly")
        url = reverse('change_password')
        data = {
            'current_password': 'testpassword123',
            'new_password': 'newpassword123'
        }
        response = api_client.put(url, data, format='json')
        assert response.status_code == status.HTTP_200_OK
        assert 'message' in response.data
        assert 'refresh' in response.data
        assert 'access' in response.data

        api_client.credentials()
        login_url = reverse('login')
        login_data = {
            'username': auth_user.username,
            'password': 'newpassword123'
        }
        login_response = api_client.post(login_url, login_data, format='json')
        assert login_response.status_code == status.HTTP_200_OK
