from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from django.db.models import Count, Sum
from django.utils import timezone
from datetime import timedelta
from .models import AdminLog, DashboardMetric
from .serializers import AdminLogSerializer, DashboardMetricSerializer
from journal.models import Journal
from subscription.models import Subscription, Plan
from django.db.models.functions import TruncDate

User = get_user_model()


class IsAdminUser(permissions.BasePermission):
    """Custom permission to only allow admin users"""

    def has_permission(self, request, view):
        return request.user and request.user.is_staff


class AdminDashboardViewSet(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]

    def _get_date_range_filters(self, days=30):
        end_date = timezone.now()
        start_date = end_date - timedelta(days=days)
        return start_date, end_date

    @action(detail=False, methods=['get'])
    def metrics(self, request):
        """Get dashboard metrics"""
        try:
            # Get date range
            start_date, end_date = self._get_date_range_filters()

            # Calculate metrics
            total_users = User.objects.count()
            active_users = User.objects.filter(
                last_login__gte=start_date).count()

            # Premium users
            premium_users = User.objects.filter(
                subscription__status='ACTIVE',
                subscription__plan__plan_type='PREMIUM'
            ).distinct().count()

            # Journal metrics
            total_journals = Journal.objects.count()
            today_journals = Journal.objects.filter(
                created_at__date=timezone.now().date()
            ).count()

            # Revenue metrics
            total_revenue = Subscription.objects.filter(
                status='ACTIVE',
                plan__plan_type='PREMIUM'
            ).aggregate(
                total=Sum('plan__price')
            )['total'] or 0

            # User growth over time
            user_growth = User.objects.filter(
                date_joined__range=(start_date, end_date)
            ).annotate(
                date=TruncDate('date_joined')
            ).values('date').annotate(
                count=Count('id')
            ).order_by('date')

            # Journal activity over time
            journal_activity = Journal.objects.filter(
                created_at__range=(start_date, end_date)
            ).annotate(
                date=TruncDate('created_at')
            ).values('date').annotate(
                count=Count('id')
            ).order_by('date')

            metrics = {
                'overview': {
                    'total_users': total_users,
                    'active_users': active_users,
                    'premium_users': premium_users,
                    'total_journals': total_journals,
                    'today_journals': today_journals,
                    'total_revenue': float(total_revenue),
                },
                'trends': {
                    'user_growth': list(user_growth),
                    'journal_activity': list(journal_activity),
                }
            }

            # Cache the metrics
            DashboardMetric.update_metric('dashboard_overview', metrics)

            return Response(metrics)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['get'])
    def user_management(self, request):
        """Get user management data"""
        try:
            users = User.objects.all().values(
                'id', 'username', 'email', 'is_active', 'date_joined',
                'is_staff', 'is_verified'
            ).order_by('-date_joined')

            return Response({
                'users': list(users)
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['get'])
    def subscription_stats(self, request):
        """Get subscription statistics"""
        try:
            # Get active subscriptions by plan
            subscriptions_by_plan = Subscription.objects.filter(
                status='ACTIVE'
            ).values(
                'plan__name'
            ).annotate(
                count=Count('id')
            )

            # Get revenue by plan
            revenue_by_plan = Subscription.objects.filter(
                status='ACTIVE'
            ).values(
                'plan__name'
            ).annotate(
                total=Sum('plan__price')
            )

            return Response({
                'subscriptions_by_plan': list(subscriptions_by_plan),
                'revenue_by_plan': list(revenue_by_plan)
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AdminLogViewSet(viewsets.ModelViewSet):
    """ViewSet for admin logs"""
    queryset = AdminLog.objects.all()
    serializer_class = AdminLogSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]

    def perform_create(self, serializer):
        """Create a new admin log entry"""
        serializer.save(
            admin_user=self.request.user,
            ip_address=self.request.META.get('REMOTE_ADDR')
        )


class UserManagementViewSet(viewsets.ModelViewSet):
    """ViewSet for user management"""
    queryset = User.objects.all()
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]

    @action(detail=True, methods=['post'])
    def toggle_active(self, request, pk=None):
        """Toggle user active status"""
        try:
            user = self.get_object()
            user.is_active = not user.is_active
            user.save()

            # Log the action
            AdminLog.objects.create(
                admin_user=request.user,
                action_type='UPDATE',
                action_detail=f"{'Activated' if user.is_active else 'Deactivated'} user account",
                target_model='User',
                target_id=user.id,
                ip_address=request.META.get('REMOTE_ADDR')
            )

            return Response({'status': 'success'})
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['post'])
    def make_admin(self, request, pk=None):
        """Make a user an admin"""
        try:
            user = self.get_object()
            if not request.user.is_superuser:
                return Response(
                    {'error': 'Only superusers can create admin users'},
                    status=status.HTTP_403_FORBIDDEN
                )

            user.is_staff = True
            user.save()

            # Log the action
            AdminLog.objects.create(
                admin_user=request.user,
                action_type='UPDATE',
                action_detail='Granted admin privileges',
                target_model='User',
                target_id=user.id,
                ip_address=request.META.get('REMOTE_ADDR')
            )

            return Response({'status': 'success'})
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
