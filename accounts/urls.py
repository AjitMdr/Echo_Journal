from django.urls import path  
from . import views


urlpatterns = [
    path('signup/initiate/',views.signup_initiate,name='signup-initiate'),
    path('signup/verify/',views.verify_otp_and_signup,name='verify-otp-and-signup'),
    path('login/',views.login,name='login'),
    path('test_token/', views.test_token,name='test-token'),
    path('forgot_password/', views.forgot_password, name='forgot-password'),
    path('resend_otp/', views.resend_otp, name='resend_otp'),
    
    path('verify-otp-reset-password/',views.verify_otp_and_reset_password, name='verify-otp-reset-password'),
    path('resend-password-reset-otp/', views.resend_password_reset_otp, name='resend-password-reset-otp')
]
