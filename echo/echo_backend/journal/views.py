from rest_framework import viewsets, status
from rest_framework.decorators import api_view, authentication_classes, permission_classes, action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.shortcuts import get_object_or_404
from django.core.exceptions import PermissionDenied
from django.contrib.auth import get_user_model
from .models import Journal
from .serializers import JournalSerializer, JournalCreateUpdateSerializer
from accounts.models import Streak, Badge, UserBadge
from mood_prediction_en.predict import predict_sentiment as predict_sentiment_en
from mood_prediction_np.predict import predict_sentiment as predict_sentiment_np
from django.utils import timezone


class JournalViewSet(viewsets.ModelViewSet):
    """
    ViewSet for handling Journal CRUD operations.

    list: Get all non-deleted journals for authenticated user
    create: Create a new journal
    retrieve: Get a specific journal
    update: Update a journal
    destroy: Soft delete a journal
    restore: Restore a soft-deleted journal
    hard_delete: Permanently delete a journal
    deleted: List all soft-deleted journals
    """
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    http_method_names = ['get', 'post', 'put', 'patch', 'delete']

    def get_queryset(self):
        """Return non-deleted journals for the authenticated user only"""
        return Journal.objects.filter(user=self.request.user).order_by('-date')

    def get_serializer_class(self):
        """Return appropriate serializer class"""
        if self.action in ['create', 'update', 'partial_update']:
            return JournalCreateUpdateSerializer
        return JournalSerializer

    def perform_create(self, serializer):
        """Create a new journal entry and update streak"""
        journal = serializer.save(user=self.request.user)

        # Update user's streak
        streak, created = Streak.objects.get_or_create(user=self.request.user)
        streak.update_streak()

        # Check and award badges based on streak
        self._check_and_award_badges(self.request.user, streak)

    def _check_and_award_badges(self, user, streak):
        """Check and award badges based on streak"""
        # Get all streak badges
        streak_badges = Badge.objects.filter(badge_type='STREAK')

        for badge in streak_badges:
            if (streak.current_streak >= badge.requirement and
                    not UserBadge.objects.filter(user=user, badge=badge).exists()):
                UserBadge.objects.create(user=user, badge=badge)

    @action(detail=True, methods=['post'])
    def restore(self, request, pk=None):
        """Restore a soft-deleted journal"""
        try:
            # Use all_objects to get even deleted journals
            journal = Journal.all_objects.get(pk=pk, user=request.user)
            if not journal.is_deleted:
                return Response({
                    'status': 'error',
                    'message': 'Journal is not deleted'
                }, status=status.HTTP_400_BAD_REQUEST)

            journal.restore()
            return Response({
                'status': 'success',
                'message': 'Journal restored successfully'
            })
        except Journal.DoesNotExist:
            return Response({
                'status': 'error',
                'message': 'Journal not found'
            }, status=status.HTTP_404_NOT_FOUND)

    @action(detail=True, methods=['delete'])
    def hard_delete(self, request, pk=None):
        """Permanently delete a journal entry"""
        try:
            journal = Journal.all_objects.get(pk=pk, user=request.user)
            journal.hard_delete()
            return Response({
                'status': 'success',
                'message': 'Journal permanently deleted'
            })
        except Journal.DoesNotExist:
            return Response({
                'status': 'error',
                'message': 'Journal not found'
            }, status=status.HTTP_404_NOT_FOUND)

    @action(detail=False, methods=['get'])
    def deleted(self, request):
        """List all soft-deleted journals"""
        deleted_journals = Journal.all_objects.filter(
            user=request.user,
            is_deleted=True
        ).order_by('-date')
        serializer = self.get_serializer(deleted_journals, many=True)
        return Response({
            'status': 'success',
            'data': serializer.data
        })

    def retrieve(self, request, *args, **kwargs):
        """Get a specific journal entry"""
        try:
            instance = self.get_object()
            serializer = self.get_serializer(instance)
            return Response({
                'status': 'success',
                'data': serializer.data
            })
        except Journal.DoesNotExist:
            return Response({
                'status': 'error',
                'message': 'Journal entry not found'
            }, status=status.HTTP_404_NOT_FOUND)
        except PermissionDenied:
            return Response({
                'status': 'error',
                'message': 'You do not have permission to access this journal'
            }, status=status.HTTP_403_FORBIDDEN)

    def list(self, request, *args, **kwargs):
        """Get all non-deleted journal entries for the user"""
        try:
            queryset = self.get_queryset()
            serializer = self.get_serializer(queryset, many=True)
            return Response({
                'status': 'success',
                'data': serializer.data
            })
        except Exception as e:
            return Response({
                'status': 'error',
                'message': 'Failed to retrieve journals',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def create(self, request, *args, **kwargs):
        """Create a new journal entry"""
        try:
            serializer = self.get_serializer(data=request.data)
            if serializer.is_valid():
                self.perform_create(serializer)
                return Response({
                    'status': 'success',
                    'message': 'Journal created successfully',
                    'data': serializer.data
                }, status=status.HTTP_201_CREATED)
            return Response({
                'status': 'error',
                'message': 'Invalid data provided',
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({
                'status': 'error',
                'message': 'Failed to create journal',
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, *args, **kwargs):
        """Update a journal entry"""
        try:
            partial = kwargs.pop('partial', False)
            instance = self.get_object()
            serializer = self.get_serializer(
                instance, data=request.data, partial=partial)
            serializer.is_valid(raise_exception=True)
            self.perform_update(serializer)
            return Response({
                'status': 'success',
                'message': 'Journal updated successfully',
                'data': serializer.data
            })
        except Exception as e:
            return Response({
                'status': 'error',
                'message': 'Failed to update journal',
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, *args, **kwargs):
        """Soft delete a journal entry"""
        try:
            print("Deleting journal")
            instance = self.get_object()
            instance.delete()  # This will call our custom soft delete method
            print("Journal soft deleted successfully")
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            print(f"Failed to delete journal: {str(e)}")
            return Response({
                'status': 'error',
                'message': 'Failed to delete journal',
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['get'])
    def analyze_sentiment(self, request, pk=None):
        """Analyze the sentiment of a journal entry"""
        try:
            journal = self.get_object()

            # Get the language from the journal
            language = journal.language or 'en'

            # Choose the appropriate sentiment analyzer based on language
            if language == 'ne':
                result = predict_sentiment_np(journal.content)
                sentiment = result.get('sentiment', 'Neutral')
                rule_based = result.get('rule_based', False)
            elif language == 'en':
                result = predict_sentiment_en(journal.content)
                # Map English emotions to simple sentiments
                emotion = result.get('sentiment', 'Neutral')
                positive_emotions = ['Happiness', 'Love']
                negative_emotions = ['Sadness', 'Anger', 'Fear']

                if emotion in positive_emotions:
                    sentiment = 'Positive'
                elif emotion in negative_emotions:
                    sentiment = 'Negative'
                else:  # Surprise or unknown
                    sentiment = 'Neutral'

                rule_based = result.get('rule_based', False)

            print(f"Sentiment: {sentiment}, Rule-based: {rule_based}")

            return Response({
                'status': 'success',
                'data': {
                    'sentiment': sentiment,
                    'rule_based': rule_based,
                    'journal_id': journal.id,
                    'title': journal.title
                }
            })
        except Exception as e:
            return Response({
                'status': 'error',
                'message': f'Failed to analyze sentiment: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['get'])
    def analyze_all_sentiments(self, request):
        """Analyze sentiments for all non-deleted journals of the user"""
        try:
            print("Analyzing all sentiments")
            # Use get_queryset() to get only non-deleted journals
            journals = self.get_queryset()
            results = []

            for journal in journals:
                language = journal.language or 'en'  # Use journal's language
                if language == 'ne':
                    result = predict_sentiment_np(journal.content)
                    sentiment = result.get('sentiment', 'Neutral')
                    rule_based = result.get('rule_based', False)
                elif language == 'en':
                    result = predict_sentiment_en(journal.content)
                    # Map English emotions to simple sentiments
                    emotion = result.get('sentiment', 'Neutral')
                    positive_emotions = ['Happiness', 'Love']
                    negative_emotions = ['Sadness', 'Anger', 'Fear']

                    if emotion in positive_emotions:
                        sentiment = 'Positive'
                    elif emotion in negative_emotions:
                        sentiment = 'Negative'
                    else:  # Surprise or unknown
                        sentiment = 'Neutral'

                    rule_based = result.get('rule_based', False)

                print(f"Sentiment: {sentiment}, Rule-based: {rule_based}")

                results.append({
                    'journal_id': journal.id,
                    'title': journal.title,
                    'date': journal.date,
                    'sentiment': sentiment,
                    'rule_based': rule_based,
                    'language': language
                })

            return Response({
                'status': 'success',
                'data': results
            })
        except Exception as e:
            return Response({
                'status': 'error',
                'message': f'Failed to analyze sentiments: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)
