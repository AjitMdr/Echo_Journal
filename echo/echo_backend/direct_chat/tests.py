import pytest
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from django.contrib.auth import get_user_model
from mixer.backend.django import mixer
from direct_chat.models import Conversation, DirectMessage

User = get_user_model()

@pytest.fixture
def api_client():
    return APIClient()

@pytest.fixture
def test_password():
    return 'strong-test-password'

@pytest.fixture
def create_user(db, test_password):
    def make_user(**kwargs):
        kwargs['password'] = test_password
        if 'username' not in kwargs:
            kwargs['username'] = mixer.faker.user_name()
        if 'email' not in kwargs:
            kwargs['email'] = mixer.faker.email()  # Ensure a unique email
        return User.objects.create_user(**kwargs)
    return make_user


@pytest.fixture
def authenticated_client(db, create_user, api_client):
    user = create_user()
    api_client.force_authenticate(user=user)
    return api_client, user

@pytest.fixture
def create_conversation(db):
    def make_conversation(participants):
        conversation = Conversation.objects.create()
        conversation.participants.add(*participants)
        return conversation
    return make_conversation

@pytest.fixture
def create_message(db):
    def make_message(**kwargs):
        return DirectMessage.objects.create(**kwargs)
    return make_message


class TestConversationViewSet:
    
    def test_list_conversations(self, authenticated_client, create_user, create_conversation):
        client, user = authenticated_client
        other_user = create_user()
        
        # Create a conversation between the users
        conversation = create_conversation([user, other_user])
        
        # Request the list of conversations
        url = reverse('conversation-list')
        response = client.get(url)
        
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 1
        assert response.data[0]['id'] == conversation.id
    
    def test_create_conversation(self, authenticated_client, create_user):
        client, user = authenticated_client
        other_user = create_user()
        
        # Create a conversation with the other user
        url = reverse('conversation-list')
        response = client.post(url, {'user_id': other_user.id})
        
        assert response.status_code == status.HTTP_201_CREATED
        assert len(response.data['participants']) == 2
        
        # Check that a duplicate conversation isn't created
        response2 = client.post(url, {'user_id': other_user.id})
        assert response2.status_code == status.HTTP_200_OK
        assert response.data['id'] == response2.data['id']
    
    def test_create_conversation_invalid_user(self, authenticated_client):
        client, _ = authenticated_client
        
        # Try to create a conversation with a non-existent user
        url = reverse('conversation-list')
        response = client.post(url, {'user_id': 9999})
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_conversation_messages(self, authenticated_client, create_user, create_conversation, create_message):
        client, user = authenticated_client
        other_user = create_user()
        
        # Create a conversation
        conversation = create_conversation([user, other_user])
        
        # Add messages to the conversation
        msg1 = create_message(
            sender=user,
            receiver=other_user,
            content="Hello",
            conversation=conversation
        )
        msg2 = create_message(
            sender=other_user,
            receiver=user,
            content="Hi there",
            conversation=conversation,
            is_read=False
        )
        
        # Get the messages
        url = reverse('conversation-messages', args=[conversation.id])
        response = client.get(url)
        
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 2
        
        # Check that messages were marked as read
        msg2.refresh_from_db()
        assert msg2.is_read is True
    
    def test_send_message(self, authenticated_client, create_user, create_conversation):
        client, user = authenticated_client
        other_user = create_user()
        
        # Create a conversation
        conversation = create_conversation([user, other_user])
        
        # Send a message
        url = reverse('conversation-send-message', args=[conversation.id])
        response = client.post(url, {'content': 'Test message'})
        
        assert response.status_code == status.HTTP_201_CREATED
        assert response.data['content'] == 'Test message'
        assert response.data['sender'] == user.id
        assert response.data['receiver'] == other_user.id
        
        # Check that conversation timestamp is updated
        conversation.refresh_from_db()
        assert conversation.updated_at is not None
    
    def test_recent_conversations(self, authenticated_client, create_user, create_conversation, create_message):
        client, user = authenticated_client
        other_user1 = create_user()
        other_user2 = create_user()
        
        # Create two conversations
        conv1 = create_conversation([user, other_user1])
        conv2 = create_conversation([user, other_user2])
        
        # Add messages to make sure they have different updated_at times
        create_message(
            sender=user,
            receiver=other_user1,
            content="Hello",
            conversation=conv1
        )
        
        import time
        time.sleep(0.1)  # Ensure different timestamps
        
        create_message(
            sender=user,
            receiver=other_user2,
            content="Hi there",
            conversation=conv2
        )
        
        # Get recent conversations
        url = reverse('conversation-recent')
        response = client.get(url)
        
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 2
        # Most recently updated conversation should be first
        assert response.data[0]['id'] == conv2.id
        assert response.data[1]['id'] == conv1.id
    
    def test_unread_count(self, authenticated_client, create_user, create_conversation, create_message):
        client, user = authenticated_client
        other_user = create_user()
        
        # Create a conversation
        conversation = create_conversation([user, other_user])
        
        # Add unread messages
        create_message(
            sender=other_user,
            receiver=user,
            content="Hello",
            conversation=conversation,
            is_read=False
        )
        create_message(
            sender=other_user,
            receiver=user,
            content="Hello again",
            conversation=conversation,
            is_read=False
        )
        
        # Get unread count
        url = reverse('conversation-unread-count')
        response = client.get(url)
        
        assert response.status_code == status.HTTP_200_OK
        assert response.data['unread_count'] == 2


class TestDirectMessageViewSet:
    
    def test_list_messages(self, authenticated_client, create_user, create_message):
        client, user = authenticated_client