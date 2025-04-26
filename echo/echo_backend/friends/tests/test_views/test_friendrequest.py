from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from django.contrib.auth import get_user_model
from friends.models import FriendRequest, Friendship

User = get_user_model()


class FriendRequestViewSetTestCase(APITestCase):
    def setUp(self):
        # Set up test users
        self.user1 = User.objects.create_user(
            username='testuser1', email='test1@example.com', password='password123'
        )
        self.user2 = User.objects.create_user(
            username='testuser2', email='test2@example.com', password='password123'
        )
        self.user3 = User.objects.create_user(
            username='testuser3', email='test3@example.com', password='password123'
        )

        # Authenticate the client with user1
        self.client.force_authenticate(user=self.user1)
        # Define the URL for listing friend requests
        self.list_url = reverse('friends:friendrequest-list')

    def test_send_friend_request(self):
        """
        Test sending a friend request to another user.
        Ensures that the friend request is successfully created.
        """
        data = {'to_user_id': self.user2.id}
        response = self.client.post(self.list_url, data, format='json')
        if response.status_code == status.HTTP_201_CREATED:
            print("test_send_friend_request: success")
        else:
            print("test_send_friend_request: not success")
        # Assert that the request was successfully created (status 201)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)


    def test_send_duplicate_friend_request(self):
        """
        Test sending a duplicate friend request.
        Ensures that sending a duplicate request results in a bad request error (400).
        """
        # Create a pending friend request from user1 to user2
        FriendRequest.objects.create(
            from_user=self.user1, to_user=self.user2, status='pending'
        )
        data = {'to_user_id': self.user2.id}
        response = self.client.post(self.list_url, data, format='json')
        if response.status_code == status.HTTP_400_BAD_REQUEST:
            print("test_send_duplicate_friend_request: success")
        else:
            print("test_send_duplicate_friend_request: not success")
        # Assert that sending a duplicate friend request results in a bad request (400)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

  

    def test_list_friend_requests(self):
        """
        Test listing all friend requests for the authenticated user.
        Ensures that the list of friend requests is returned correctly.
        """
        # Create several pending friend requests
        FriendRequest.objects.create(
            from_user=self.user1, to_user=self.user2, status='pending'
        )
        FriendRequest.objects.create(
            from_user=self.user3, to_user=self.user1, status='pending'
        )
        FriendRequest.objects.create(
            from_user=self.user2, to_user=self.user3, status='pending'
        )

        response = self.client.get(self.list_url)
        if response.status_code == status.HTTP_200_OK:
            print("test_list_friend_requests: success")
        else:
            print("test_list_friend_requests: not success")
        # Assert that the list of friend requests is returned successfully (status 200)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_accept_friend_request(self):
        """
        Test accepting a pending friend request.
        Ensures that the friend request is successfully accepted.
        """
        # Create a pending friend request from user2 to user1
        friend_request = FriendRequest.objects.create(
            from_user=self.user2, to_user=self.user1, status='pending'
        )

        # Define the URL for accepting the friend request
        accept_url = reverse('friends:friendrequest-accept', kwargs={'pk': friend_request.id})
        response = self.client.post(accept_url)
        if response.status_code == status.HTTP_200_OK:
            print("test_accept_friend_request: success")
        else:
            print("test_accept_friend_request: not success")
        # Assert that the friend request is accepted successfully (status 200)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_reject_friend_request(self):
        """
        Test rejecting a pending friend request.
        Ensures that the friend request is successfully rejected.
        """
        # Create a pending friend request from user2 to user1
        friend_request = FriendRequest.objects.create(
            from_user=self.user2, to_user=self.user1, status='pending'
        )

        # Define the URL for rejecting the friend request
        reject_url = reverse('friends:friendrequest-reject', kwargs={'pk': friend_request.id})
        response = self.client.post(reject_url)
        if response.status_code == status.HTTP_200_OK:
            print("test_reject_friend_request: success")
        else:
            print("test_reject_friend_request: not success")
        # Assert that the friend request is rejected successfully (status 200)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

   




    def test_cannot_accept_other_user_request(self):
        """
        Test that a user cannot accept a friend request sent to another user.
        Ensures that trying to accept a request that was not sent to the authenticated user results in a not found error (404).
        """
        # Create a pending friend request from user2 to user3
        friend_request = FriendRequest.objects.create(
            from_user=self.user2, to_user=self.user3, status='pending'
        )

        # Define the URL for accepting the friend request
        accept_url = reverse('friends:friendrequest-accept', kwargs={'pk': friend_request.id})
        response = self.client.post(accept_url)
        if response.status_code == status.HTTP_404_NOT_FOUND:
            print("test_cannot_accept_other_user_request: success")
        else:
            print("test_cannot_accept_other_user_request: not success")
        # Assert that a user cannot accept a request that was sent to another user (status 404)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
