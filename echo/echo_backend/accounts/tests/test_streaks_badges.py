# tests/test_streak_badges.py
import pytest
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import timedelta
from accounts.models import Streak, Badge, UserBadge

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

@pytest.fixture
def create_streak(auth_user):
    def _create_streak(current_streak=0, longest_streak=0, last_journal_date=None):
        streak, created = Streak.objects.get_or_create(
            user=auth_user,
            defaults={
                'current_streak': current_streak,
                'longest_streak': longest_streak,
                'last_journal_date': last_journal_date
            }
        )
        if not created:
            streak.current_streak = current_streak
            streak.longest_streak = longest_streak
            streak.last_journal_date = last_journal_date
            streak.save()
        return streak
    return _create_streak

@pytest.fixture
def create_badge():
    def _create_badge(name="Test Badge", description="Test Description", 
                       badge_type="STREAK", icon="🔥", requirement=5):
        badge = Badge.objects.create(
            name=name,
            description=description,
            badge_type=badge_type,
            icon=icon,
            requirement=requirement
        )
        return badge
    return _create_badge

@pytest.mark.django_db
class TestStreakManagement:
    """Tests for the streak management endpoints"""

    def test_get_current_streak(self, api_client, auth_user, create_streak):
        # Test: retrieving the current streak returns correct values
        yesterday = timezone.now() - timedelta(days=1)
        create_streak(current_streak=5, longest_streak=10, last_journal_date=yesterday)
        url = reverse('streak-current-streak')
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data['current_streak'] == 5
        assert response.data['longest_streak'] == 10
        assert 'streak_emoji' in response.data

    def test_streak_broken(self, api_client, auth_user, create_streak):
        # Test: if more than one day has passed, streak resets to 0
        two_days_ago = timezone.now() - timedelta(days=2)
        streak = create_streak(current_streak=5, longest_streak=10, last_journal_date=two_days_ago)
        url = reverse('streak-current-streak')
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data['current_streak'] == 0
        streak.refresh_from_db()
        assert streak.current_streak == 0

    def test_streak_continues(self, api_client, auth_user, create_streak):
        # Test: if only one day has passed, streak remains unchanged
        yesterday = timezone.now() - timedelta(days=1)
        streak = create_streak(current_streak=5, longest_streak=10, last_journal_date=yesterday)
        url = reverse('streak-current-streak')
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data['current_streak'] == 5

@pytest.mark.django_db
class TestLeaderboard:
    """Tests for the leaderboard endpoint"""

    def test_leaderboard(self, api_client, auth_user, create_user):
        # Test: users are ranked by current streak descending
        user1 = create_user(username="user1", email="user1@example.com")
        user2 = create_user(username="user2", email="user2@example.com")
        user3 = create_user(username="user3", email="user3@example.com")
        Streak.objects.create(user=auth_user, current_streak=5)
        Streak.objects.create(user=user1, current_streak=10)
        Streak.objects.create(user=user2, current_streak=15)
        Streak.objects.create(user=user3, current_streak=3)
        url = reverse('streak-leaderboard')
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        results = response.data['results']
        assert len(results) == 4
        assert results[0]['user_id'] == user2.id
        assert results[1]['user_id'] == user1.id
        assert results[2]['user_id'] == auth_user.id
        assert results[3]['user_id'] == user3.id

    def test_leaderboard_pagination(self, api_client, auth_user):
        # Test: leaderboard pagination returns correct page size and metadata
        for i in range(20):
            user = User.objects.create_user(
                username=f"user{i}",
                email=f"user{i}@example.com",
                password="password",
                is_verified=True
            )
            Streak.objects.create(user=user, current_streak=i+1)
        url = reverse('streak-leaderboard') + "?page=1&page_size=10"
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data['results']) == 10
        assert response.data['total_count'] == 20
        assert response.data['current_page'] == 1
        assert response.data['total_pages'] == 2
        url = reverse('streak-leaderboard') + "?page=2&page_size=10"
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data['results']) == 10
        assert response.data['current_page'] == 2

@pytest.mark.django_db
class TestBadges:
    """Tests for badge-related endpoints"""

    def test_get_user_badges(self, api_client, auth_user, create_badge):
        # Test: retrieving badges returns those awarded to the user
        badge = create_badge(name="5-Day Streak", requirement=5)
        UserBadge.objects.create(user=auth_user, badge=badge)
        url = reverse('badge-user-badges')
        response = api_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 1
        assert response.data[0]['badge']['name'] == "5-Day Streak"

    def test_check_and_award_badges(self, api_client, auth_user, create_badge, create_streak):
        # Test: awarding badges based on current streak thresholds
        badge1 = create_badge(name="5-Day Streak", requirement=5)
        badge2 = create_badge(name="10-Day Streak", requirement=10)
        streak = create_streak(current_streak=7)
        url = reverse('badge-check-and-award-badges')
        response = api_client.post(url)
        assert response.status_code == status.HTTP_200_OK
        user_badges = UserBadge.objects.filter(user=auth_user)
        assert user_badges.count() == 1
        assert user_badges.first().badge.id == badge1.id
        streak.current_streak = 12
        streak.save()
        response = api_client.post(url)
        assert response.status_code == status.HTTP_200_OK
        user_badges = UserBadge.objects.filter(user=auth_user)
        assert user_badges.count() == 2
        badge_ids = set(ub.badge.id for ub in user_badges)
        assert badge_ids == {badge1.id, badge2.id}
