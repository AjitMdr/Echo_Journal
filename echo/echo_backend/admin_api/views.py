from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from django.db.models import Count, Sum, Q, F, ExpressionWrapper, FloatField
from django.utils import timezone
from datetime import timedelta
from .models import AdminLog, DashboardMetric
from .serializers import AdminLogSerializer, DashboardMetricSerializer
from journal.models import Journal
from subscription.models import Subscription, Plan, Payment
from django.db.models.functions import TruncDate, TruncMonth, TruncWeek

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
        """Get comprehensive dashboard metrics"""
        try:
            # Get date range
            start_date, end_date = self._get_date_range_filters()
            current_date = timezone.now()

            # User metrics
            total_users = User.objects.count()

            # Active users - users who have created journals in last 30 days
            active_users = User.objects.filter(
                journals__date__gte=start_date
            ).distinct().count()

            new_users = User.objects.filter(
                date_joined__range=(start_date, end_date)
            ).count()

            # Premium users (excluding expired subscriptions)
            premium_users = User.objects.filter(
                subscription__status='ACTIVE',
                subscription__plan__plan_type='PREMIUM',
                subscription__end_date__gt=current_date
            ).distinct().count()

            # Subscription metrics (excluding expired)
            active_subscriptions = Subscription.objects.filter(
                status='ACTIVE',
                end_date__gt=current_date
            ).count()

            # Revenue metrics (only from active subscriptions)
            total_revenue = Subscription.objects.filter(
                status='ACTIVE',
                plan__plan_type='PREMIUM',
                end_date__gt=current_date
            ).aggregate(
                total=Sum('plan__price')
            )['total'] or 0

            monthly_revenue = Subscription.objects.filter(
                status='ACTIVE',
                plan__plan_type='PREMIUM',
                created_at__month=current_date.month,
                end_date__gt=current_date
            ).aggregate(
                total=Sum('plan__price')
            )['total'] or 0

            # Journal metrics - count all journals
            total_journals = Journal.objects.count()

            # Get today's date range
            today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = today_start + timedelta(days=1)

            today_journals = Journal.objects.filter(
                date__gte=today_start,
                date__lt=today_end
            ).count()

            # Calculate average journals per user excluding users with 0 journals
            users_with_journals = User.objects.annotate(
                journal_count=Count('journals')
            ).filter(journal_count__gt=0)

            total_journals_by_active_users = sum(
                user.journal_count for user in users_with_journals
            )

            avg_journals_per_user = (
                total_journals_by_active_users / users_with_journals.count()
                if users_with_journals.count() > 0 else 0
            )

            # User engagement metrics
            active_users_ratio = (
                active_users / total_users * 100) if total_users > 0 else 0
            premium_conversion_rate = (
                premium_users / total_users * 100) if total_users > 0 else 0

            metrics = {
                'overview': {
                    'total_users': total_users,
                    'active_users': active_users,
                    'new_users': new_users,
                    'premium_users': premium_users,
                    'active_subscriptions': active_subscriptions,
                    'total_journals': total_journals,
                    'today_journals': today_journals,
                    'total_revenue': float(total_revenue),
                    'monthly_revenue': float(monthly_revenue),
                    'avg_journals_per_user': round(avg_journals_per_user, 2),
                    'active_users_ratio': round(active_users_ratio, 2),
                    'premium_conversion_rate': round(premium_conversion_rate, 2),
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
    def trends(self, request):
        """Get trend data for charts"""
        try:
            start_date, end_date = self._get_date_range_filters()

            # User growth trend - daily new users
            user_growth = User.objects.filter(
                date_joined__range=(start_date, end_date)
            ).annotate(
                trend_date=TruncDate('date_joined')
            ).values('trend_date').annotate(
                count=Count('id')
            ).order_by('trend_date')

            # Journal activity trend - daily journal entries
            journal_activity = Journal.objects.filter(
                date__range=(start_date, end_date)
            ).annotate(
                trend_date=TruncDate('date')
            ).values('trend_date').annotate(
                count=Count('id')
            ).order_by('trend_date')

            # Revenue trend - daily new subscription revenue
            revenue_trend = Subscription.objects.filter(
                status='ACTIVE',
                created_at__range=(start_date, end_date)
            ).annotate(
                trend_date=TruncDate('created_at')
            ).values('trend_date').annotate(
                revenue=Sum('plan__price')
            ).order_by('trend_date')

            # User engagement trend - daily active users
            engagement_trend = Journal.objects.filter(
                date__range=(start_date, end_date)
            ).annotate(
                trend_date=TruncDate('date')
            ).values('trend_date').annotate(
                count=Count('user_id', distinct=True)
            ).order_by('trend_date')

            trends = {
                'user_growth': list(user_growth),
                'journal_activity': list(journal_activity),
                'revenue_trend': list(revenue_trend),
                'engagement_trend': list(engagement_trend)
            }

            return Response(trends)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['get'])
    def subscription_analytics(self, request):
        """Get detailed subscription analytics"""
        try:
            current_date = timezone.now()
            start_date = current_date - timedelta(days=30)

            # Get all active subscriptions (status='ACTIVE')
            active_subs = Subscription.objects.filter(status='ACTIVE')

            # Get plan types and their counts
            plan_distribution = []

            # Count Free Plan users (active subscriptions with plan_id=5)
            free_plan = active_subs.filter(plan_id=5)
            free_plan_count = free_plan.count()
            if free_plan_count > 0:
                plan_distribution.append({
                    'plan__name': 'Free Plan',
                    'count': free_plan_count,
                    'revenue': 0.0
                })

            # Count Premium Plan users (active subscriptions with plan_id=6)
            premium_plan = active_subs.filter(plan_id=6)
            premium_count = premium_plan.count()
            if premium_count > 0:
                premium_revenue = premium_plan.aggregate(
                    total=Sum('plan__price'))['total'] or 0.0
                plan_distribution.append({
                    'plan__name': 'Premium Plan',
                    'count': premium_count,
                    'revenue': float(premium_revenue)
                })

            # Monthly subscription growth - group by creation month
            monthly_growth = Subscription.objects.annotate(
                month=TruncMonth('created_at')
            ).values('month').annotate(
                count=Count('id')
            ).order_by('month')

            # Calculate subscription retention metrics
            total_subs = Subscription.objects.count()
            active_subs_count = active_subs.count()

            # Calculate churn rate (opposite of retention)
            churned_subs = Subscription.objects.filter(
                status='EXPIRED').count()
            churn_rate = (churned_subs / total_subs *
                          100) if total_subs > 0 else 0

            # Calculate renewal rate (for subscriptions that were due for renewal in last 30 days)
            subs_due_renewal = Subscription.objects.filter(
                end_date__range=(start_date, current_date)
            ).count()

            renewed_subs = Subscription.objects.filter(
                end_date__range=(start_date, current_date),
                status='ACTIVE'
            ).count()

            renewal_rate = (renewed_subs / subs_due_renewal *
                            100) if subs_due_renewal > 0 else 0

            analytics = {
                'plan_distribution': plan_distribution,
                'monthly_growth': list(monthly_growth),
                'retention_metrics': {
                    'total_subscriptions': total_subs,
                    'active_subscriptions': active_subs_count,
                    'expired_subscriptions': churned_subs,
                    'retention_rate': round((active_subs_count / total_subs * 100) if total_subs > 0 else 0, 2),
                    'churn_rate': round(churn_rate, 2),
                    'renewal_rate': round(renewal_rate, 2)
                }
            }

            return Response(analytics)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['get'])
    def user_analytics(self, request):
        """Get detailed user analytics"""
        try:
            start_date, end_date = self._get_date_range_filters()

            # Get total users for retention calculation
            total_users = User.objects.count()

            # Get users who joined before the 30-day period
            old_users = User.objects.filter(date_joined__lt=start_date).count()

            # User activity by time of day
            activity_by_hour = Journal.objects.filter(
                date__range=(start_date, end_date)
            ).annotate(
                hour=F('date__hour')
            ).values('hour').annotate(
                count=Count('id')
            ).order_by('hour')

            # User engagement by day of week
            engagement_by_day = Journal.objects.filter(
                date__range=(start_date, end_date)
            ).annotate(
                day_of_week=F('date__week_day')
            ).values('day_of_week').annotate(
                count=Count('id')
            ).order_by('day_of_week')

            # User retention metrics
            returning_users = User.objects.filter(
                last_login__range=(start_date, end_date),
                date_joined__lt=start_date
            ).count()

            # Active users who created journals in last 30 days
            active_users = User.objects.filter(
                journals__date__range=(start_date, end_date)
            ).distinct().count()

            analytics = {
                'activity_by_hour': list(activity_by_hour),
                'engagement_by_day': list(engagement_by_day),
                'retention_metrics': {
                    'total_users': total_users,
                    'old_users': old_users,  # Users who joined before 30 days ago
                    'returning_users': returning_users,  # Old users who logged in recently
                    'active_users': active_users,  # Users who created journals recently
                    'login_retention_rate': round((returning_users / old_users * 100) if old_users > 0 else 0, 2),
                    'engagement_rate': round((active_users / total_users * 100) if total_users > 0 else 0, 2)
                }
            }

            return Response(analytics)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['get'])
    def user_management(self, request):
        """Get user management data"""
        try:
            users = User.objects.all().order_by('-date_joined')

            # Process users and their profile pictures
            processed_users = []
            for user in users:
                user_data = {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'is_active': user.is_active,
                    'date_joined': user.date_joined,
                    'is_staff': user.is_staff,
                    'is_verified': user.is_verified,
                    'profile_picture': request.build_absolute_uri(user.profile_picture.url) if user.profile_picture else None
                }
                processed_users.append(user_data)

            return Response({
                'users': processed_users
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
            current_date = timezone.now()

            # Get active subscriptions by plan (excluding expired)
            subscriptions_by_plan = Subscription.objects.filter(
                status='ACTIVE',
                end_date__gt=current_date
            ).values(
                'plan__name'
            ).annotate(
                count=Count('id')
            )

            # Get revenue by plan (excluding expired)
            revenue_by_plan = Subscription.objects.filter(
                status='ACTIVE',
                end_date__gt=current_date
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

    @action(detail=False, methods=['GET'])
    def transactions(self, request):
        """
        Get all subscription transactions with summary metrics
        """
        try:
            # Get all payments
            payments = Payment.objects.select_related(
                'subscription__user', 'subscription__plan').all().order_by('-payment_date')

            # Calculate summary metrics
            total_revenue = sum(payment.amount for payment in payments)
            total_transactions = payments.count()
            active_subscriptions = Subscription.objects.filter(
                status='ACTIVE').count()

            # Format payment data
            transactions = []
            for payment in payments:
                transactions.append({
                    'id': payment.id,
                    'user': {
                        'id': payment.subscription.user.id,
                        'email': payment.subscription.user.email,
                        'name': f"{payment.subscription.user.first_name} {payment.subscription.user.last_name}",
                    },
                    'plan': payment.subscription.plan.name,
                    'amount': payment.amount,
                    'status': payment.status,  # Use payment status instead of subscription status
                    'transaction_date': payment.payment_date,
                })

            return Response({
                'summary': {
                    'total_revenue': total_revenue,
                    'total_transactions': total_transactions,
                    'active_subscriptions': active_subscriptions,
                },
                'transactions': transactions,
            })
        except Exception as e:
            return Response({'error': str(e)}, status=500)


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
