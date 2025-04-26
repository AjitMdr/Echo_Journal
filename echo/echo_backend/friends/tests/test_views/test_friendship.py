# tests/test_views/test_friendship.py
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from django.contrib.auth import get_user_model
from friends.models import Friendship

User = get_user_model()


# tests/test_views/test_friendship.py
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from django.contrib.auth import get_user_model
from friends.models import Friendship

User = get_user_model()


class FriendshipViewSetTestCase(APITestCase):
    def setUp(self):
        # Create test users and authenticate one of them for testing
        self.user1 = User.objects.create_user(
            username='testuser1', email='test1@example.com', password='password123'
        )
        self.user2 = User.objects.create_user(
            username='testuser2', email='test2@example.com', password='password123'
        )
        self.user3 = User.objects.create_user(
            username='testuser3', email='test3@example.com', password='password123'
        )
        
        self.client.force_authenticate(user=self.user1)
        
        # Create friendships to be used in the test
        self.friendship1 = Friendship.objects.create(user1=self.user1, user2=self.user2)
        self.friendship2 = Friendship.objects.create(user1=self.user2, user2=self.user3)
        
        # Define all relevant URLs
        self.list_url = reverse('friends:friendship-list')
        self.detail_url = reverse('friends:friendship-detail', kwargs={'pk': self.friendship1.id})
        self.unfriend_url = reverse('friends:friendship-unfriend', kwargs={'pk': self.friendship1.id})
        self.other_friendship_url = reverse('friends:friendship-detail', kwargs={'pk': self.friendship2.id})
        self.other_unfriend_url = reverse('friends:friendship-unfriend', kwargs={'pk': self.friendship2.id})
    
    # This test is done to verify that the authenticated user can list only their own friendships
    def test_list_friendships(self):
        response = self.client.get(self.list_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['id'], self.friendship1.id)
    
    # This test is done to check if the user can retrieve a specific friendship involving them
    def test_get_friendship_detail(self):
        response = self.client.get(self.detail_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['id'], self.friendship1.id)
    
    # This test is done to check if the user can unfriend (delete a friendship) successfully
    def test_unfriend(self):
        response = self.client.delete(self.unfriend_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(Friendship.objects.count(), 1)
        self.assertFalse(Friendship.objects.filter(id=self.friendship1.id).exists())
    
    