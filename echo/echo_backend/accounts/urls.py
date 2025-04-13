from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .views import StreakViewSet, BadgeViewSet, UserViewSet
from rest_framework_simplejwt.views import TokenRefreshView, TokenVerifyView

router = DefaultRouter()
router.register(r'streaks', StreakViewSet, basename='streak')
router.register(r'badges', BadgeViewSet, basename='badge')
router.register(r'users', UserViewSet, basename='user')

urlpatterns = [
    path('register/', views.register, name='register'),
    path('signup/verify/', views.verify_otp_and_signup,
         name='verify-otp-and-signup'),
    path('login/', views.login, name='login'),
    path('login/2fa/verify/', views.verify_2fa_login, name='verify-2fa-login'),
    path('login/2fa/resend/', views.resend_2fa_login_otp,
         name='resend-2fa-login-otp'),
    path('forgot_password/', views.forgot_password, name='forgot-password'),
    path('resend_otp/', views.resend_otp, name='resend_otp'),
    path('verify-otp-reset-password/', views.verify_otp_and_reset_password,
         name='verify-otp-reset-password'),
    path('resend-password-reset-otp/', views.resend_password_reset_otp,
         name='resend-password-reset-otp'),
    path('profile/', views.get_profile, name='get_profile'),
    path('profile/update/', views.update_profile, name='update_profile'),
    path('profile/password/', views.change_password, name='change_password'),
    path('profile/picture/', views.update_profile_picture,
         name='update_profile_picture'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('verify-account/', views.verify_account_request, name='verify-account'),
    path('2fa/status/', views.get_two_factor_status, name='two-factor-status'),
    path('2fa/toggle/', views.toggle_two_factor, name='two-factor-toggle'),
    path('', include(router.urls)),
]
