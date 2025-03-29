from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/', include('accounts.urls')),
    path('api/journal/', include('journal.urls')),
    path('api/friends/', include('friends.urls')),
    path('api/chat/', include('direct_chat.urls')),
    path('api/subscription/', include('subscription.urls')),
] 