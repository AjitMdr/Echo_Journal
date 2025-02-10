from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

from .serializers import UserSerializer

from django.db import transaction, IntegrityError
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.core.cache import cache
import logging

"""imports for otp """

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db import IntegrityError, transaction, DatabaseError
from rest_framework.authtoken.models import Token
import logging

from django.core.cache import cache

import random
from django.core.mail import send_mail
from django.conf import settings
from django.core.cache import cache 

from rest_framework.permissions import AllowAny

from django.contrib.auth import authenticate
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework.authtoken.models import Token
from .serializers import UserSerializer


@api_view(['POST'])
# def signup(request):
#     serializer = UserSerializer(data=request.data)
#     if serializer.is_valid():
#         serializer.save()
#         user = User.objects.get(username=request.data['username'])
#         user.set_password(request.data['password'])
#         user.save()
#         token = Token.objects.create(user=user)
#         return Response({'token': token.key, 'user': serializer.data}, status=status.HTTP_201_CREATED)
#     return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



@csrf_exempt
def login(request):
    """
    Login view that authenticates user and returns token
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        import json
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return JsonResponse(
                {'error': 'Username and password are required'},
                status=400
            )

        user = authenticate(request, username=username, password=password)

        if user is None:
            return JsonResponse(
                {'error': 'Invalid credentials'},
                status=400
            )

        if not user.is_active:
            return JsonResponse(
                {'error': 'This account is inactive'},
                status=400
            )

        # Get or create token
        token, _ = Token.objects.get_or_create(user=user)
        
        # Serialize user data
        serializer = UserSerializer(user)

        return JsonResponse({
            'token': token.key,
            'user': serializer.data,
            'message': 'Login successful'
        })

    except json.JSONDecodeError:
        return JsonResponse(
            {'error': 'Invalid JSON format'},
            status=400
        )
    except Exception as e:
        return JsonResponse({
            'error': 'An error occurred during login',
            'details': str(e)
        }, status=500)

@api_view(['GET'])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def test_token(request):
    return Response("passed!")

logger = logging.getLogger(__name__)
User = get_user_model()
# ---------- UTILITIES ----------

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

# ---------- SIGNUP INITIATE ----------

@api_view(['POST'])
@permission_classes([AllowAny])
def signup_initiate(request):
    try:
        required_fields = ['username', 'password', 'email']
        if not all(field in request.data for field in required_fields):
            return Response(
                {'error': 'Missing required fields'},
                status=status.HTTP_400_BAD_REQUEST
            )

        email = request.data['email']
        username = request.data['username']

        #  Check for existing email and username
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

        # Generate OTP
        otp = generate_otp()

        # Store OTP and user data in cache
        cache_key = f"signup_otp_{email}"
        cache_data = {
            'otp': otp,
            'username': username,
            'password': request.data['password']
        }
        cache.set(cache_key, cache_data, timeout=600)  # OTP valid for 10 minutes

        # Send OTP via email
        if not send_otp_email(email, otp):
            return Response(
                {'error': 'Failed to send OTP email'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response({
            'message': 'OTP sent successfully',
            'email': email
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error during signup initiation: {str(e)}")
        return Response(
            {
                'error': 'An error occurred during signup initiation',
                'details': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# ---------- VERIFY OTP & SIGNUP ----------

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

        cache_key = f"signup_otp_{email}"
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

        # Create user in transaction
        try:
            with transaction.atomic():
                user = User(
                    username=cached_data['username'],
                    email=email
                )
                user.set_password(cached_data['password'])
                user.save()

                token = Token.objects.create(user=user)
                cache.delete(cache_key)  # Clear OTP cache

                return Response({
                    'token': token.key,
                    'user': {
                        'id': user.id,
                        'username': user.username,
                        'email': user.email
                    },
                    'message': 'Signup completed successfully'
                }, status=status.HTTP_201_CREATED)

        except IntegrityError as e:
            logger.error(f"Integrity error during signup: {e}")
            return Response(
                {'error': 'Failed to create user account'},
                status=status.HTTP_400_BAD_REQUEST
            )

    except Exception as e:
        logger.error(f"Error during signup completion: {str(e)}")
        return Response(
            {
                'error': 'An error occurred during signup completion',
                'details': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )