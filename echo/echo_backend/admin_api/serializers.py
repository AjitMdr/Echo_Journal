from rest_framework import serializers
from .models import AdminLog, DashboardMetric
from django.contrib.auth import get_user_model

User = get_user_model()


class AdminLogSerializer(serializers.ModelSerializer):
    admin_username = serializers.CharField(
        source='admin_user.username', read_only=True)

    class Meta:
        model = AdminLog
        fields = [
            'id',
            'admin_user',
            'admin_username',
            'action_type',
            'action_detail',
            'target_model',
            'target_id',
            'created_at',
            'ip_address',
        ]
        read_only_fields = ['admin_user', 'created_at', 'ip_address']


class DashboardMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = DashboardMetric
        fields = ['metric_name', 'metric_value', 'last_updated']
        read_only_fields = ['last_updated']


class AdminUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'is_active',
            'is_staff',
            'is_superuser',
            'date_joined',
            'last_login',
        ]
        read_only_fields = ['date_joined', 'last_login']
        extra_kwargs = {
            'password': {'write_only': True}
        }
