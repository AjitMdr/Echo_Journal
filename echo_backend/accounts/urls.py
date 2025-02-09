from django.urls import path  
from . import views

urlpatterns = [
    path('signup/initiate/',views.signup_initiate,name='signup-initiate'),
    path('signup/verify/',views.verify_otp_and_signup,name='verify-otp-and-signup'),
    path('login/',views.login,name='login'),
    path('test_token/', views.test_token,name='test-token'),
]
