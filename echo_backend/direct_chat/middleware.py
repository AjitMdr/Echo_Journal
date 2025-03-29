from channels.middleware import BaseMiddleware
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from urllib.parse import parse_qs
import logging

logger = logging.getLogger(__name__)
User = get_user_model()


@database_sync_to_async
def get_user_from_token(token_key):
    try:
        # Remove 'Bearer ' prefix if present
        if token_key.startswith('Bearer '):
            token_key = token_key[7:]

        logger.info(f"Attempting to validate token: {token_key[:10]}...")
        access_token = AccessToken(token_key)
        user_id = access_token['user_id']
        user = User.objects.get(id=user_id)
        logger.info(f"Successfully validated token for user: {user_id}")
        return user
    except InvalidToken as e:
        logger.error(f"Invalid token error: {str(e)}")
        return AnonymousUser()
    except TokenError as e:
        logger.error(f"Token error: {str(e)}")
        return AnonymousUser()
    except User.DoesNotExist as e:
        logger.error(f"User not found: {str(e)}")
        return AnonymousUser()
    except Exception as e:
        logger.error(f"Unexpected error in get_user_from_token: {str(e)}")
        return AnonymousUser()


class JWTAuthMiddleware(BaseMiddleware):
    async def __call__(self, scope, receive, send):
        token = None

        # Try to get token from query string first
        query_string = scope.get('query_string', b'').decode()
        query_params = parse_qs(query_string)
        token = query_params.get('token', [None])[0]

        # If no token in query string, check headers
        if not token:
            headers = dict(scope.get('headers', []))
            auth_header = headers.get(b'authorization', b'').decode()
            if auth_header.startswith('Bearer '):
                token = auth_header

        if token:
            logger.info("Token found, attempting to authenticate user")
            # Get the user from the token
            scope['user'] = await get_user_from_token(token)
        else:
            logger.warning("No token found in request")
            scope['user'] = AnonymousUser()

        return await super().__call__(scope, receive, send)
