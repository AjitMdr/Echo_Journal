from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from django.contrib.auth import get_user_model
from friends.models import FriendRequest, Friendship

User = get_user_model()


class UserSearchViewSetTestCase(APITestCase):
    def setUp(self):
        # Create test users
        self.user1 = User.objects.create_user(
            username='testuser1', email='test1@example.com', 
            first_name='Test', last_name='User1', password='password123'
        )
        self.user2 = User.objects.create_user(
            username='testuser2', email='test2@example.com', 
            first_name='Test', last_name='User2', password='password123'
        )
        self.user3 = User.objects.create_user(
            username='otheruser', email='other@example.com', 
            first_name='Other', last_name='User', password='password123'
        )
        
        # Authenticate user1
        self.client.force_authenticate(user=self.user1)
        
        # Create friendship between user1 and user2
        self.friendship = Friendship.objects.create(user1=self.user1, user2=self.user2)
        
        # Create pending friend request from user1 to user3
        self.friend_request = FriendRequest.objects.create(
            from_user=self.user1, to_user=self.user3, status='pending'
        )
        
        # URL for user search
        self.search_url = reverse('friends:usersearch-list')

    # This test is done to verify that users are returned with correct friendship statuses (friend/pending/etc.)
    def test_search_users_with_friendship_status(self):
        response = self.client.get(f"{self.search_url}?search=test")
        if response.status_code == status.HTTP_200_OK:
            print("test_search_users_with_friendship_status: success")
        else:
            print("test_search_users_with_friendship_status: not success")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        results = response.data['results']
        usernames = [user['username'] for user in results]
        self.assertIn('testuser2', usernames)
        
        for user in results:
            if user['username'] == 'testuser2':
                self.assertEqual(user['friendship_status'], 'friend')
            elif user['username'] == 'otheruser':
                self.assertEqual(user['friendship_status'], 'pending_sent')

    # This test is done to verify that an empty search query returns all users except the current user
    def test_search_users_with_empty_query(self):
        response = self.client.get(self.search_url)
        if response.status_code == status.HTTP_200_OK:
            print("test_search_users_with_empty_query: success")
        else:
            print("test_search_users_with_empty_query: not success")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        results = response.data['results']
        self.assertEqual(len(results), 2)
        usernames = [user['username'] for user in results]
        self.assertIn('testuser2', usernames)
        self.assertIn('otheruser', usernames)
        self.assertNotIn('testuser1', usernames)

    # This test is done to verify that search results are filtered based on the given query string
    def test_search_users_with_specific_query(self):
        response = self.client.get(f"{self.search_url}?search=other")
        if response.status_code == status.HTTP_200_OK:
            print("test_search_users_with_specific_query: success")
        else:
            print("test_search_users_with_specific_query: not success")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        results = response.data['results']
        usernames = [user['username'] for user in results]
        self.assertIn('otheruser', usernames)
        self.assertNotIn('testuser2', usernames)

    # This test is done to verify that pagination works as expected when search results exceed the page size
    def test_search_users_with_pagination(self):
        for i in range(12):
            User.objects.create_user(
                username=f'paginationuser{i}',
                email=f'pagination{i}@example.com',
                password='password123'
            )
            
        response = self.client.get(f"{self.search_url}?page=2")
        if response.status_code == status.HTTP_200_OK:
            print("test_search_users_with_pagination: success")
        else:
            print("test_search_users_with_pagination: not success")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        self.assertIn('count', response.data)
        self.assertIn('next', response.data)
        self.assertIn('previous', response.data)
        self.assertEqual(len(response.data['results']), 4)  # 14 total users (excluding user1), 10 on page 1, 4 on page 2
