import pytest
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from django.utils import timezone
from django.contrib.auth import get_user_model
from journal.models import Journal
from accounts.models import Streak, Badge, UserBadge
from unittest.mock import patch

# Define test fixtures
User = get_user_model()

@pytest.fixture
def api_client():
    return APIClient()

@pytest.fixture
def test_user():
    user = User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='testpassword123'
    )
    return user

@pytest.fixture
def auth_client(api_client, test_user):
    from rest_framework_simplejwt.tokens import RefreshToken
    refresh = RefreshToken.for_user(test_user)
    api_client.credentials(HTTP_AUTHORIZATION=f'Bearer {refresh.access_token}')
    return api_client

@pytest.fixture
def test_journal(test_user):
    journal = Journal.objects.create(
        user=test_user,
        title="Test Journal Entry",
        content="This is a test journal entry with sufficient content for testing.",
        language="en"
    )
    return journal

@pytest.fixture
def journal_data():
    return {
        "title": "New Journal Entry",
        "content": "This is a new journal entry with enough content to pass validation.",
        "language": "en"
    }

# Test class for journal API scenarios
@pytest.mark.django_db
class TestJournalAPI:

    # Test scenario: Authenticated users can list journals
    def test_list_journals_authenticated(self, auth_client, test_journal):
        # This test checks if an authenticated user can retrieve the list of journals
        url = reverse('journal-list')
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data['status'] == 'success'
        assert len(response.data['data']) == 1
        assert response.data['data'][0]['title'] == test_journal.title

    # Test scenario: Unauthenticated users cannot list journals
    def test_list_journals_unauthenticated(self, api_client):
        # This test checks if unauthenticated users are blocked from listing journals
        url = reverse('journal-list')
        response = api_client.get(url)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    # Test scenario: Authenticated users can create a journal
    def test_create_journal(self, auth_client, journal_data):
        # This test ensures that authenticated users can create a new journal entry
        url = reverse('journal-list')
        response = auth_client.post(url, journal_data, format='json')
        assert response.status_code == status.HTTP_201_CREATED
        assert response.data['status'] == 'success'
        assert response.data['data']['title'] == journal_data['title']
        assert Journal.objects.filter(title=journal_data['title']).exists()

    # Test scenario: Invalid data results in a 400 Bad Request
    def test_create_journal_invalid_data(self, auth_client):
        # This test ensures that invalid data (e.g., too short title/content) results in a 400 error
        url = reverse('journal-list')
        invalid_data = {
            "title": "A", 
            "content": "Too short", 
            "language": "invalid"
        }
        response = auth_client.post(url, invalid_data, format='json')
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.data['status'] == 'error'

    # Test scenario: Authenticated users can retrieve a specific journal by ID
    def test_retrieve_journal(self, auth_client, test_journal):
        # This test checks if an authenticated user can retrieve a specific journal entry by ID
        url = reverse('journal-detail', args=[test_journal.id])
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data['status'] == 'success'
        assert response.data['data']['id'] == test_journal.id
        assert response.data['data']['title'] == test_journal.title

    # Test scenario: Trying to retrieve a nonexistent journal returns 404
    def test_retrieve_nonexistent_journal(self, auth_client):
        # This test ensures that trying to retrieve a journal that does not exist results in a 404 error
        url = reverse('journal-detail', args=[999])
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_404_NOT_FOUND

    # Test scenario: Authenticated users can update an existing journal
    def test_update_journal(self, auth_client, test_journal):
        # This test checks if an authenticated user can update an existing journal entry
        url = reverse('journal-detail', args=[test_journal.id])
        update_data = {
            "title": "Updated Journal Title",
            "content": "This is the updated content for the journal entry.",
            "language": "en"
        }
        response = auth_client.put(url, update_data, format='json')
        assert response.status_code == status.HTTP_200_OK
        assert response.data['status'] == 'success'
        assert response.data['data']['title'] == update_data['title']
        test_journal.refresh_from_db()
        assert test_journal.title == update_data['title']

    # Test scenario: Authenticated users can partially update a journal
    def test_partial_update_journal(self, auth_client, test_journal):
        # This test checks if an authenticated user can partially update a journal entry
        url = reverse('journal-detail', args=[test_journal.id])
        update_data = {
            "title": "Partially Updated Title"
        }
        response = auth_client.patch(url, update_data, format='json')
        assert response.status_code == status.HTTP_200_OK
        assert response.data['status'] == 'success'
        assert response.data['data']['title'] == update_data['title']
        assert response.data['data']['content'] == test_journal.content

    # Test scenario: Authenticated users can delete a journal
    def test_delete_journal(self, auth_client, test_journal):
        # This test checks if an authenticated user can delete an existing journal entry
        url = reverse('journal-detail', args=[test_journal.id])
        response = auth_client.delete(url)
        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert not Journal.objects.filter(id=test_journal.id).exists()
        assert Journal.all_objects.filter(id=test_journal.id, is_deleted=True).exists()

   

    # Test scenario: Sentiment analysis for an English journal
    @patch('journal.views.predict_sentiment_en')
    def test_analyze_sentiment_en(self, mock_predict, auth_client, test_journal):
        # This test checks if sentiment analysis works for English language journals
        mock_predict.return_value = {'sentiment': 'Happiness', 'rule_based': False}
        url = reverse('journal-analyze-sentiment', args=[test_journal.id])
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data['status'] == 'success'
        assert response.data['data']['sentiment'] == 'Positive'

    # Test scenario: Sentiment analysis for a Nepali journal
    @patch('journal.views.predict_sentiment_np')
    def test_analyze_sentiment_np(self, mock_predict, auth_client, test_journal):
        # This test checks if sentiment analysis works for Nepali language journals
        test_journal.language = 'ne'
        test_journal.save()
        mock_predict.return_value = {'sentiment': 'Positive', 'rule_based': True}
        url = reverse('journal-analyze-sentiment', args=[test_journal.id])
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data['status'] == 'success'
        assert response.data['data']['sentiment'] == 'Positive'

    # Test scenario: Streak update when creating a new journal
    def test_streak_update_on_journal_creation(self, auth_client, journal_data, test_user):
        # This test checks if the streak is updated when a new journal is created
        streak = Streak.objects.create(
            user=test_user,
            current_streak=5,
            longest_streak=5,
            last_journal_date=timezone.now() - timezone.timedelta(days=1)
        )
        url = reverse('journal-list')
        response = auth_client.post(url, journal_data, format='json')
        assert response.status_code == status.HTTP_201_CREATED
        streak.refresh_from_db()
        assert streak.current_streak == 6
        assert streak.longest_streak == 6
        assert streak.last_journal_date.date() == timezone.now().date()
