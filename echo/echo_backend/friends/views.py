from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.pagination import PageNumberPagination
from django.db.models import Q
from django.contrib.auth import get_user_model
from .models import FriendRequest, Friendship
from .serializers import FriendRequestSerializer, FriendshipSerializer, UserSearchSerializer

User = get_user_model()


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100


class UserSearchViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = UserSearchSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    pagination_class = StandardResultsSetPagination
    filter_backends = [filters.SearchFilter]
    search_fields = ['username', 'email', 'first_name', 'last_name']

    def get_queryset(self):
        """Return all users with their friendship status"""
        search_query = self.request.query_params.get('search', '')
        current_user = self.request.user
        print(f"🔍 Searching for users with query: {search_query}")
        print(
            f"👤 Current user: {current_user.username} (ID: {current_user.id})")

        # Get all users except current user
        queryset = User.objects.exclude(id=current_user.id)

        # Get current user's friends
        friends = Friendship.objects.filter(
            Q(user1=current_user) | Q(user2=current_user)
        ).values_list('user1', 'user2')

        # Create sets for efficient lookup
        friend_ids = {
            user_id for pair in friends for user_id in pair if user_id != current_user.id}
        print(f"👥 Friend IDs: {friend_ids}")

        # Get pending friend requests
        pending_sent = set(FriendRequest.objects.filter(
            from_user=current_user,
            status='pending'
        ).values_list('to_user_id', flat=True))
        print(f"📤 Pending sent requests to: {pending_sent}")

        pending_received = set(FriendRequest.objects.filter(
            to_user=current_user,
            status='pending'
        ).values_list('from_user_id', flat=True))
        print(f"📥 Pending received requests from: {pending_received}")

        # Apply search filter if query exists
        if search_query:
            queryset = queryset.filter(
                Q(username__icontains=search_query) |
                Q(email__icontains=search_query) |
                Q(first_name__icontains=search_query) |
                Q(last_name__icontains=search_query)
            )

        # Store friendship information in class instance
        self.friend_ids = friend_ids
        self.pending_sent = pending_sent
        self.pending_received = pending_received

        return queryset

    def paginate_queryset(self, queryset):
        """Override to annotate friendship status after pagination"""
        page = super().paginate_queryset(queryset)
        if page is not None:
            # Annotate friendship status for paginated results
            for user in page:
                if user.id in self.friend_ids:
                    user.friendship_status = 'friend'
                elif user.id in self.pending_sent:
                    user.friendship_status = 'pending_sent'
                elif user.id in self.pending_received:
                    user.friendship_status = 'pending_received'
                else:
                    user.friendship_status = 'none'

                print(
                    f"👤 User {user.username} (ID: {user.id}) - Status: {user.friendship_status}")
                print(f"  - Is friend: {user.id in self.friend_ids}")
                print(f"  - Has sent request: {user.id in self.pending_sent}")
                print(
                    f"  - Has received request: {user.id in self.pending_received}")

        return page


class FriendRequestViewSet(viewsets.ModelViewSet):
    serializer_class = FriendRequestSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        """Return friend requests for the authenticated user"""
        return FriendRequest.objects.filter(
            Q(from_user=self.request.user) | Q(to_user=self.request.user)
        ).select_related('from_user', 'to_user')

    def perform_create(self, serializer):
        """Create a new friend request"""
        print(f"Creating friend request with data: {self.request.data}")
        print(
            f"Current user: {self.request.user.username} (ID: {self.request.user.id})")
        try:
            serializer.save(from_user=self.request.user)
            print("Friend request created successfully")
        except Exception as e:
            print(f"Error creating friend request: {str(e)}")
            raise

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

    @action(detail=True, methods=['post'])
    def accept(self, request, pk=None):
        """Accept a friend request"""
        friend_request = self.get_object()

        # Check if the request is pending and the current user is the recipient
        if friend_request.status != 'pending':
            return Response({
                'status': 'error',
                'message': 'This friend request cannot be accepted'
            }, status=status.HTTP_400_BAD_REQUEST)

        if friend_request.to_user != request.user:
            return Response({
                'status': 'error',
                'message': 'You cannot accept this friend request'
            }, status=status.HTTP_403_FORBIDDEN)

        # Create friendship
        Friendship.objects.create(
            user1=friend_request.from_user,
            user2=friend_request.to_user
        )

        # Delete the friend request after creating friendship
        friend_request.delete()

        return Response({
            'status': 'success',
            'message': 'Friend request accepted and friendship created'
        })

    @action(detail=True, methods=['post'])
    def reject(self, request, pk=None):
        """Reject a friend request"""
        friend_request = self.get_object()

        # Check if the request is pending and the current user is the recipient
        if friend_request.status != 'pending':
            return Response({
                'status': 'error',
                'message': 'This friend request cannot be rejected'
            }, status=status.HTTP_400_BAD_REQUEST)

        if friend_request.to_user != request.user:
            return Response({
                'status': 'error',
                'message': 'You cannot reject this friend request'
            }, status=status.HTTP_403_FORBIDDEN)

        # Update request status
        friend_request.status = 'rejected'
        friend_request.save()

        return Response({
            'status': 'success',
            'message': 'Friend request rejected'
        })


class FriendshipViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = FriendshipSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        """Return friendships for the authenticated user"""
        return Friendship.objects.filter(
            Q(user1=self.request.user) | Q(user2=self.request.user)
        )

    @action(detail=True, methods=['delete'])
    def unfriend(self, request, pk=None):
        """Remove a friendship"""
        print(f"🔄 Unfriend request received for friendship ID: {pk}")
        print(
            f"👤 Current user: {request.user.username} (ID: {request.user.id})")

        try:
            friendship = self.get_object()
            print(
                f"👥 Friendship: {friendship.user1.username} - {friendship.user2.username}")

            # Check if the current user is part of the friendship
            if request.user not in [friendship.user1, friendship.user2]:
                print(
                    f"❌ User {request.user.username} is not part of this friendship")
                return Response({
                    'status': 'error',
                    'message': 'You cannot remove this friendship'
                }, status=status.HTTP_403_FORBIDDEN)

            friendship.delete()
            print(f"✅ Friendship successfully deleted")
            return Response({
                'status': 'success',
                'message': 'Friendship removed successfully'
            })
        except Exception as e:
            print(f"❌ Error unfriending: {str(e)}")
            return Response({
                'status': 'error',
                'message': f'Failed to unfriend: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)
