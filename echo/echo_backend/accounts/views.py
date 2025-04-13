from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from django.db import transaction, IntegrityError
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.core.cache import cache
import logging

# """imports for otp """

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import get_user_model

from django.db import IntegrityError, transaction, DatabaseError
from rest_framework.authtoken.models import Token


import random
from django.core.mail import send_mail
from django.conf import settings


from rest_framework.permissions import AllowAny

from django.contrib.auth import authenticate
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework.authtoken.models import Token
from .serializers import UserLoginSerializer

from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode

# imports for forgot
from django.core.mail import send_mail
from django.conf import settings
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.utils import timezone
from datetime import timedelta

from django.contrib.auth.models import User

# for reset
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth.models import User

from django.contrib.auth.hashers import make_password
from django.contrib.auth.hashers import check_password

from accounts.serializers import UserProfileUpdateSerializer

from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.decorators import api_view, parser_classes
from django.conf import settings


# login api


from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework.parsers import JSONParser
from django.contrib.auth import get_user_model

from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import UserSerializer
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.core.management.utils import get_random_secret_key

from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from .models import Streak, Badge, UserBadge
from .serializers import StreakSerializer, BadgeSerializer, UserBadgeSerializer


@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    serializer = UserLoginSerializer(data=request.data)

    if serializer.is_valid():
        user = serializer.validated_data['user']

        # Check if 2FA is enabled
        if user.two_factor_enabled:
            # Generate and send OTP
            otp = generate_otp()
            cache_key = f"2fa_login_otp_{user.email}"

            # Store OTP in cache with user ID
            cache.set(cache_key, {
                'otp': otp,
                'user_id': user.id
            }, timeout=300)  # 5 minutes expiry

            # Send OTP via email
            if not send_2fa_login_otp(user.email, otp):
                return Response(
                    {'error': 'Failed to send 2FA code'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            return Response({
                'requires_2fa': True,
                'email': user.email,
                'message': '2FA code has been sent to your email'
            }, status=status.HTTP_200_OK)

        # If 2FA is not enabled, proceed with normal login
        refresh = RefreshToken.for_user(user)
        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': {
                'username': user.username,
                'email': user.email,
                'id': user.id,
                'is_verified': user.is_verified,
                'role': user.role
            }
        }, status=status.HTTP_200_OK)

    # Handle validation errors
    error_message = serializer.errors.get(
        'non_field_errors', ['Invalid credentials'])[0]
    if 'verify your account' in error_message.lower():
        return Response({
            'error': error_message,
            'needs_verification': True,
            'email': request.data.get('username')
        }, status=status.HTTP_403_FORBIDDEN)

    return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)


def send_2fa_login_otp(email, otp):
    """Send 2FA OTP for login"""
    subject = 'Login Verification Code'
    message = f'Your verification code for login is: {otp}\nThis code is valid for 5 minutes.'

    try:
        send_mail(
            subject,
            message,
            settings.EMAIL_HOST_USER,
            [email],
            fail_silently=False,
        )
        return True
    except Exception as e:
        logger.error(f"Error sending 2FA login OTP to {email}: {e}")
        return False


@api_view(['POST'])
@permission_classes([AllowAny])
def verify_2fa_login(request):
    """Verify 2FA code during login"""
    try:
        email = request.data.get('email')
        otp = request.data.get('otp')

        if not email or not otp:
            return Response(
                {'error': 'Email and verification code are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Get OTP from cache
        cache_key = f"2fa_login_otp_{email}"
        cached_data = cache.get(cache_key)

        if not cached_data:
            return Response(
                {'error': 'Verification code has expired. Please login again.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if str(otp) != str(cached_data['otp']):
            return Response(
                {'error': 'Invalid verification code'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user = User.objects.get(id=cached_data['user_id'])
            refresh = RefreshToken.for_user(user)

            # Clear the OTP cache
            cache.delete(cache_key)

            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'user': {
                    'username': user.username,
                    'email': user.email,
                    'id': user.id,
                    'is_verified': user.is_verified
                }
            }, status=status.HTTP_200_OK)

        except User.DoesNotExist:
            return Response(
                {'error': 'User not found'},
                status=status.HTTP_404_NOT_FOUND
            )

    except Exception as e:
        logger.error(f"Error during 2FA verification: {str(e)}")
        return Response(
            {'error': 'An error occurred during verification'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def resend_2fa_login_otp(request):
    """Resend 2FA code during login"""
    try:
        email = request.data.get('email')

        if not email:
            return Response(
                {'error': 'Email is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        cache_key = f"2fa_login_otp_{email}"
        cached_data = cache.get(cache_key)

        if not cached_data:
            return Response(
                {'error': 'No active login session found. Please login again.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Generate new OTP
        new_otp = generate_otp()
        cached_data['otp'] = new_otp
        # Reset timeout to 5 minutes
        cache.set(cache_key, cached_data, timeout=300)

        # Send new OTP
        if not send_2fa_login_otp(email, new_otp):
            return Response(
                {'error': 'Failed to send verification code'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response({
            'message': 'New verification code sent successfully',
            'email': email
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error during 2FA code resend: {str(e)}")
        return Response(
            {'error': 'An error occurred while resending verification code'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# function to create otp and send it


def generate_otp():
    """Generate a 6-digit OTP"""
    return str(random.randint(100000, 999999))


def send_otp_email(email, otp):
    """Send OTP to user's email"""
    subject = 'Email Verification OTP'
    message = f'Your OTP for email verification is: {otp}\nThis OTP is valid for 10 minutes.'

    try:
        send_mail(
            subject,
            message,
            settings.EMAIL_HOST_USER,
            [email],
            fail_silently=False,
        )
        return True
    except Exception as e:
        logger.error(f"Error sending OTP email to {email}: {e}")
        return False

# sign up initiation


logger = logging.getLogger(__name__)


@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    try:
        required_fields = ['username', 'password', 'email']
        if not all(field in request.data for field in required_fields):
            return Response(
                {'error': 'Missing required fields'},
                status=status.HTTP_400_BAD_REQUEST
            )

        email = request.data['email']
        username = request.data['username']
        password = request.data['password']

        # Check if email or username is already used
        if User.objects.filter(email=email).exists():
            return Response(
                {'error': 'Email is already in use.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if User.objects.filter(username=username).exists():
            return Response(
                {'error': 'Username is already taken.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Create user with is_verified=False
        try:
            with transaction.atomic():
                user = User(
                    username=username,
                    email=email,
                    is_verified=False
                )
                user.set_password(password)
                user.save()

                # Generate OTP
                otp = generate_otp()

                # Store OTP in cache
                cache_key = f"verify_email_otp_{email}"
                cache.set(cache_key, {
                    'otp': otp,
                    'user_id': user.id
                }, timeout=600)  # 10 minutes expiry

                # Send OTP via email
                if not send_otp_email(email, otp):
                    # If email fails, rollback transaction
                    raise Exception('Failed to send OTP email')

                return Response({
                    'message': 'Registration successful! Please verify your email with the OTP sent.',
                    'email': email
                }, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Error creating user or sending OTP: {e}")
            return Response(
                {'error': 'Failed to complete registration'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        return Response(
            {
                'error': 'An error occurred during registration',
                'details': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# verify sign up


@api_view(['POST'])
@permission_classes([AllowAny])
def verify_otp_and_signup(request):
    try:
        email = request.data.get('email')
        otp = request.data.get('otp')

        if not email or not otp:
            return Response(
                {'error': 'Email and OTP are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Get OTP from cache
        cache_key = f"verify_email_otp_{email}"
        cached_data = cache.get(cache_key)

        if not cached_data:
            return Response(
                {'error': 'OTP has expired. Please request a new one.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if otp != cached_data['otp']:
            return Response(
                {'error': 'Invalid OTP'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            with transaction.atomic():
                user = User.objects.get(id=cached_data['user_id'])
                user.is_verified = True
                user.save()

                # Create auth token
                token = Token.objects.create(user=user)

                cache.delete(cache_key)  # Clear OTP cache

                return Response({
                    'token': token.key,
                    'user': {
                        'id': user.id,
                        'username': user.username,
                        'email': user.email
                    },
                    'message': 'Email verified successfully'
                }, status=status.HTTP_200_OK)

        except User.DoesNotExist:
            return Response(
                {'error': 'User not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except IntegrityError as e:
            logger.error(f"Database error during verification: {e}")
            return Response(
                {'error': 'Database error occurred'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    except Exception as e:
        logger.error(f"Error during OTP verification: {str(e)}")
        return Response(
            {'error': 'An error occurred during verification'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# resent otp
@api_view(['POST'])
@permission_classes([AllowAny])
def resend_otp(request):
    try:
        email = request.data.get('email')

        if not email:
            return Response(
                {'error': 'Email is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        cache_key = f"signup_otp_{email}"
        cached_data = cache.get(cache_key)

        if not cached_data:
            return Response(
                {'error': 'No OTP request found for this email. Please initiate signup again.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Generate new OTP
        new_otp = generate_otp()
        cached_data['otp'] = new_otp  # Update cached OTP
        cache.set(cache_key, cached_data, timeout=600)  # Reset OTP timeout

        # Send new OTP via email
        if not send_otp_email(email, new_otp):
            return Response(
                {'error': 'Failed to send OTP email'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response({
            'message': 'New OTP sent successfully',
            'email': email
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error during OTP resend: {str(e)}")
        return Response(
            {
                'error': 'An error occurred while resending OTP',
                'details': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# forgot password api
@api_view(['POST'])  # Add GET if you need GET requests too
@permission_classes([AllowAny])
def forgot_password(request):
    """
    Forgot password view that sends OTP
    """
    if request.method != 'POST':
        return Response(
            {'error': 'Method not allowed'},
            status=status.HTTP_405_METHOD_NOT_ALLOWED
        )

    try:
        data = request.data
        email = data.get('email')

        if not email:
            return Response(
                {'error': 'Email is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Check if user exists
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response(
                {'error': 'No account found with this email'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Generate OTP
        otp = generate_otp()

        # Store OTP in cache
        cache_key = f"password_reset_otp_{email}"
        cache.set(cache_key, {
            'otp': otp,
            'user_id': user.id
        }, timeout=600)  # 10 minutes expiry

        # Send OTP via email
        if not send_password_reset_otp_email(email, otp):
            return Response(
                {'error': 'Failed to send OTP email'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response({
            'message': 'OTP has been sent to your email',
            'email': email
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error during password reset request: {str(e)}")
        return Response(
            {
                'error': 'An error occurred while processing the request',
                'details': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# send mail function


def send_password_reset_otp_email(email, otp):
    """Send OTP for password reset to user's email"""
    subject = 'Password Reset OTP'
    message = f'Your OTP for password reset is: {otp}\nThis OTP is valid for 10 minutes.'

    try:
        send_mail(
            subject,
            message,
            settings.EMAIL_HOST_USER,
            [email],
            fail_silently=False,
        )
        return True
    except Exception as e:
        logger.error(f"Error sending password reset OTP email to {email}: {e}")
        return False

# verify otp api


@api_view(['POST'])
@permission_classes([AllowAny])
def verify_otp_and_reset_password(request):
    try:
        email = request.data.get('email')
        otp = request.data.get('otp')
        new_password = request.data.get('new_password')

        if not all([email, otp, new_password]):
            return Response(
                {'error': 'Email, OTP, and new password are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Verify OTP
        cache_key = f"password_reset_otp_{email}"
        cached_data = cache.get(cache_key)

        if not cached_data:
            return Response(
                {'error': 'OTP expired or invalid'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if str(otp) != str(cached_data['otp']):
            return Response(
                {'error': 'Invalid OTP'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user = User.objects.get(id=cached_data['user_id'])
        except User.DoesNotExist:
            return Response(
                {'error': 'User not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Set new password
        user.set_password(new_password)
        user.save()

        # Clear the OTP cache
        cache.delete(cache_key)

        return Response(
            {'message': 'Password has been reset successfully'},
            status=status.HTTP_200_OK
        )

    except Exception as e:
        logger.error(f"Error during password reset: {str(e)}")
        return Response(
            {'error': 'An error occurred during password reset',
                'details': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# resend otp api


@api_view(['POST'])
@permission_classes([AllowAny])
def resend_password_reset_otp(request):
    try:
        email = request.data.get('email')

        if not email:
            return Response(
                {'error': 'Email is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response(
                {'error': 'No account found with this email'},
                status=status.HTTP_404_NOT_FOUND
            )

        cache_key = f"password_reset_otp_{email}"
        cached_data = cache.get(cache_key)

        if not cached_data:
            return Response(
                {'error': 'No active password reset request found. Please initiate password reset again.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Generate new OTP
        new_otp = generate_otp()
        cached_data['otp'] = new_otp
        cache.set(cache_key, cached_data, timeout=600)  # Reset timeout

        # Send new OTP
        if not send_password_reset_otp_email(email, new_otp):
            return Response(
                {'error': 'Failed to send OTP email'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response({
            'message': 'New OTP sent successfully',
            'email': email
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error during OTP resend: {str(e)}")
        return Response(
            {
                'error': 'An error occurred while resending OTP',
                'details': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


User = get_user_model()


@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def get_profile(request):
    """Get user profile information"""
    serializer = UserSerializer(request.user)
    return Response(serializer.data)


@api_view(['PUT'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def update_profile(request):
    """Update user profile information"""
    serializer = UserProfileUpdateSerializer(
        request.user, data=request.data, partial=True)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['PUT'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def change_password(request):
    """Change user password"""
    current_password = request.data.get('current_password')
    new_password = request.data.get('new_password')

    if not current_password or not new_password:
        return Response({'error': 'Both current and new passwords are required'}, status=status.HTTP_400_BAD_REQUEST)

    user = request.user
    if not check_password(current_password, user.password):
        return Response({'error': 'Current password is incorrect'}, status=status.HTTP_400_BAD_REQUEST)

    user.set_password(new_password)
    user.save()
    return Response({'message': 'Password updated successfully'})


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser])
def update_profile_picture(request):
    """Update user profile picture"""
    user = request.user

    # Debugging - print user model class and available fields
    print(f"User model: {user.__class__.__name__}")
    print(f"User model fields: {[f.name for f in user._meta.fields]}")

    if 'profile_picture' not in request.FILES:
        return Response({'error': 'No image file provided'}, status=status.HTTP_400_BAD_REQUEST)

    # Save new profile picture
    user.profile_picture = request.FILES['profile_picture']

    try:
        user.save()
        serializer = UserProfileUpdateSerializer(user)
        return Response({
            'message': 'Profile picture updated successfully',
            'data': serializer.data
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            'error': f'Error saving profile picture: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class StreakViewSet(viewsets.ModelViewSet):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    serializer_class = StreakSerializer

    def get_queryset(self):
        return Streak.objects.filter(user=self.request.user)

    def _get_streak_response(self, streak):
        """Helper method to get consistent streak response"""
        # Check if streak is current, if not, reset it
        if not streak.is_streak_current():
            if streak.current_streak > 0:  # Only reset if there was an existing streak
                streak.current_streak = 0  # Reset to 0 for broken streaks
                streak.save()
        elif streak.current_streak == 0:  # If streak is current but showing 0, set it to 1
            streak.current_streak = 1
            streak.save()

        emoji = self._get_streak_emoji(streak.current_streak)
        print(f"Debug - Raw emoji: {emoji}")

        response_data = {
            'current_streak': streak.current_streak,
            'longest_streak': streak.longest_streak,
            'last_journal_date': streak.last_journal_date,
            'emoji': emoji
        }
        print(f"Debug - Full response data: {response_data}")
        return response_data

    @action(detail=False, methods=['get'])
    def current_streak(self, request):
        streak, created = Streak.objects.get_or_create(user=request.user)
        return Response(self._get_streak_response(streak))

    @action(detail=False, methods=['get'], url_path='user-streak')
    def user_streak(self, request):
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({'error': 'user_id is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(id=user_id)
            streak, created = Streak.objects.get_or_create(user=user)
            return Response(self._get_streak_response(streak))
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

    def _get_streak_emoji(self, streak_count):
        """Return appropriate emoji based on streak count"""
        if streak_count == 0:
            return '💫'  # Using raw emoji
        elif streak_count < 7:
            return '🔥'
        elif streak_count < 30:
            return '⚡'
        elif streak_count < 100:
            return '🌟'
        else:
            return '👑'


class BadgeViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    serializer_class = BadgeSerializer

    def get_queryset(self):
        return Badge.objects.all()

    @action(detail=False, methods=['get'])
    def user_badges(self, request):
        user_badges = UserBadge.objects.filter(user=request.user)
        serializer = UserBadgeSerializer(user_badges, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def check_and_award_badges(self, request):
        user = request.user
        streak = Streak.objects.get(user=user)

        # Get all streak badges
        streak_badges = Badge.objects.filter(badge_type='STREAK')

        for badge in streak_badges:
            if (streak.current_streak >= badge.requirement and
                    not UserBadge.objects.filter(user=user, badge=badge).exists()):
                UserBadge.objects.create(user=user, badge=badge)

        return Response({'status': 'success'})


class UserViewSet(viewsets.ModelViewSet):
    """ViewSet for user management"""
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Get queryset based on user"""
        if self.action == 'list':
            return User.objects.all()
        return User.objects.filter(id=self.request.user.id)

    @action(detail=False, methods=['get'])
    def profile(self, request):
        """Get current user's profile information"""
        user = request.user
        serializer = self.get_serializer(user)
        return Response(serializer.data)


@api_view(['POST'])
@permission_classes([AllowAny])
def verify_account_request(request):
    """
    Endpoint to verify account credentials and send OTP
    """
    try:
        username = request.data.get('username')
        password = request.data.get('password')

        if not username or not password:
            return Response(
                {'error': 'Username and password are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response(
                {'error': 'User does not exist'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Verify password
        if not user.check_password(password):
            return Response(
                {'error': 'Invalid credentials'},
                status=status.HTTP_401_UNAUTHORIZED
            )

        # Check if account is already verified
        if user.is_verified:
            return Response(
                {'error': 'Account is already verified'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Generate and send OTP
        otp = generate_otp()
        cache_key = f"verify_email_otp_{user.email}"

        # Store OTP in cache
        cache.set(cache_key, {
            'otp': otp,
            'user_id': user.id
        }, timeout=600)  # 10 minutes expiry

        # Send OTP via email
        if not send_otp_email(user.email, otp):
            return Response(
                {'error': 'Failed to send OTP email'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response({
            'message': 'OTP sent successfully',
            'email': user.email
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error during account verification request: {str(e)}")
        return Response(
            {'error': 'An error occurred during verification'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_two_factor_status(request):
    """Get the current 2FA status for the user"""
    try:
        return Response({
            'is_enabled': request.user.two_factor_enabled
        }, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error getting 2FA status: {str(e)}")
        return Response(
            {'error': 'Failed to get 2FA status'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def toggle_two_factor(request):
    """Enable or disable 2FA for the user"""
    try:
        enable = request.data.get('enable')
        if enable is None:
            return Response(
                {'error': 'Enable parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = request.user
        user.two_factor_enabled = enable
        user.save()

        return Response({
            'message': '2FA settings updated successfully',
            'is_enabled': user.two_factor_enabled
        }, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error updating 2FA status: {str(e)}")
        return Response(
            {'error': 'Failed to update 2FA settings'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
