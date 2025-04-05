from django.contrib import admin
from django.urls import path, include
from rest_framework.authtoken import views as auth_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/', include('accounts.urls')),
    path('api/auth/subscription/', include('subscription.urls')),
    path('api/journal/', include('journal.urls')),
    path('api/friends/', include('friends.urls')),
    path('api/direct-chat/', include('direct_chat.urls')),
    # For token authentication
    path('api-token-auth/', auth_views.obtain_auth_token),
]
