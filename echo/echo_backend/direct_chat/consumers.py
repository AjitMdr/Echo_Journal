import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from .models import DirectMessage, Conversation
from datetime import datetime
import traceback
import logging
from django.db.models import Q

# Set up logger
logger = logging.getLogger('direct_chat')

User = get_user_model()


class DirectChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        try:
            logger.info('⭐ WebSocket connect attempt')
            logger.info(f'⭐ Connection scope: {self.scope}')
            logger.info(f'⭐ Headers: {dict(self.scope["headers"])}')

            # 1. Verify authentication
            self.user = self.scope['user']
            logger.info(f'⭐ User from scope: {self.user}')
            
            if not self.user or not self.user.is_authenticated:
                logger.error('❌ User is not authenticated')
                await self.close(code=4003)
                return

            # 2. Get friend ID from URL
            try:
                self.friend_id = self.scope['url_route']['kwargs']['friend_id']
                logger.info(f'⭐ Found friend_id in URL: {self.friend_id}')

                # Verify it's a valid numeric ID
                if not self.friend_id.isdigit():
                    logger.error(f'❌ Friend ID is not numeric: {self.friend_id}')
                    await self.close(code=4004)
                    return
            except (KeyError, AttributeError) as e:
                logger.error(f'❌ Failed to get friend ID from URL: {e}')
                logger.error(f'❌ URL route: {self.scope.get("url_route")}')
                await self.close(code=4004)
                return

            # 3. Verify the friend exists
            try:
                friend = await self.get_user(self.friend_id)
                if not friend:
                    logger.error(f'❌ Friend with ID {self.friend_id} does not exist')
                    await self.close(code=4004)
                    return
                self.friend = friend
                logger.info(f"⭐ Friend validated: {friend.username} (ID: {friend.id})")
            except Exception as e:
                logger.error(f'❌ Error finding friend: {e}')
                await self.close(code=4004)
                return

            # 4. Accept the connection
            await self.accept()
            logger.info(f"✅ WebSocket connection accepted for user {self.user.id} and friend {self.friend_id}")

            # 5. Get or create conversation and load message history
            conversation = await self.get_or_create_conversation(self.friend_id)
            logger.info(f"✅ Got conversation: {conversation.id}")
            
            message_history = await self.get_message_history(conversation)
            logger.info(f"✅ Loaded message history: {len(message_history)} messages")
            
            # 6. Send message history
            history_data = {
                'type': 'chat_message',
                'messages': message_history
            }
            logger.info(f"✅ Sending message history: {json.dumps(history_data)[:200]}...")
            await self.send(text_data=json.dumps(history_data))
            logger.info("✅ Sent message history to client")

            # 7. Send a confirmation message
            await self.send(text_data=json.dumps({
                'type': 'connection_established',
                'user_id': str(self.user.id),
                'friend_id': self.friend_id,
                'timestamp': datetime.now().isoformat()
            }))

        except Exception as e:
            logger.error(f"❌ Error during WebSocket connection: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            await self.close(code=4000)

    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnected: Code={close_code}")

    async def receive(self, text_data):
        try:
            # Parse the incoming data
            text_data_json = json.loads(text_data)
            logger.info(f"Received message: {text_data_json}")

            # Handle ping messages
            if text_data_json.get('type') == 'ping':
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }))
                return

            # Handle regular messages
            message_content = text_data_json.get('message', '').strip()
            if not message_content:
                logger.warning("Empty message received, ignoring")
                return

            # Get or create conversation
            conversation = await self.get_or_create_conversation(self.friend_id)

            # Save message to database
            direct_message = await self.save_message(
                sender=self.user,
                receiver=self.friend,
                content=message_content,
                conversation=conversation
            )

            # Send message back to the sender for confirmation
            await self.send(text_data=json.dumps({
                'id': str(direct_message.id),
                'message': message_content,
                'content': message_content,
                'sender': str(self.user.id),
                'sender_id': str(self.user.id),
                'sender_username': self.user.username,
                'receiver': self.friend_id,
                'receiver_id': self.friend_id,
                'receiver_username': self.friend.username,
                'timestamp': direct_message.timestamp.isoformat(),
                'is_read': False,
                'conversation_id': str(conversation.id),
                'type': 'chat_message'
            }))

        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
        except Exception as e:
            logger.error(f"Error in receive: {e}")
            logger.error(traceback.format_exc())
            await self.send(text_data=json.dumps({
                'type': 'error',
                'error': str(e)
            }))

    @database_sync_to_async
    def get_user(self, user_id):
        try:
            return User.objects.get(id=user_id)
        except User.DoesNotExist:
            return None
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None

    @database_sync_to_async
    def get_or_create_conversation(self, friend_id):
        try:
            friend = User.objects.get(id=friend_id)
            conversations = Conversation.objects.filter(
                participants=self.user).filter(participants=friend)

            if conversations.exists():
                return conversations.first()

            conversation = Conversation.objects.create()
            conversation.participants.add(self.user, friend)
            return conversation
        except User.DoesNotExist:
            raise ValueError(f"User with ID {friend_id} does not exist")
        except Exception as e:
            raise ValueError(f"Error creating conversation: {e}")

    @database_sync_to_async
    def save_message(self, sender, receiver, content, conversation=None):
        try:
            # Get or create conversation if not provided
            if not conversation:
                friend = User.objects.get(id=self.friend_id)
                conversations = Conversation.objects.filter(
                    participants=self.user).filter(participants=friend)
                if conversations.exists():
                    conversation = conversations.first()
                else:
                    conversation = Conversation.objects.create()
                    conversation.participants.add(self.user, friend)

            # Create and save the message
            message = DirectMessage.objects.create(
                sender=sender,
                receiver=receiver,
                content=content,
                conversation=conversation
            )
            
            # Update conversation timestamp
            conversation.save()
            
            return message
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            raise ValueError(f"Error saving message: {e}")

    @database_sync_to_async
    def get_message_history(self, conversation):
        """Get message history between users"""
        try:
            # Query messages using conversation ID
            messages = DirectMessage.objects.filter(
                conversation=conversation
            ).order_by('timestamp')
            
            logger.info(f"Found {messages.count()} messages in history for conversation {conversation.id}")
            
            formatted_messages = []
            for msg in messages:
                message_data = {
                    'id': str(msg.id),
                    'message': msg.content,
                    'content': msg.content,
                    'sender': str(msg.sender.id),
                    'sender_id': str(msg.sender.id),
                    'sender_username': msg.sender.username,
                    'receiver': str(msg.receiver.id),
                    'receiver_id': str(msg.receiver.id),
                    'receiver_username': msg.receiver.username,
                    'timestamp': msg.timestamp.isoformat(),
                    'is_read': msg.is_read,
                    'conversation_id': str(msg.conversation.id),
                    'type': 'chat_message'
                }
                formatted_messages.append(message_data)
            
            logger.info(f"Formatted {len(formatted_messages)} messages")
            if formatted_messages:
                logger.info(f"Sample message: {json.dumps(formatted_messages[0])}")
                
            return formatted_messages
            
        except Exception as e:
            logger.error(f"Error getting message history: {e}")
            logger.error(traceback.format_exc())
            return []
