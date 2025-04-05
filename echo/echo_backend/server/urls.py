from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/', include('accounts.urls')),
    path('api/journal/', include('journal.urls')),
    path('api/friends/', include('friends.urls')),
    path('api/direct-chat/', include('direct_chat.urls')),
    path('api/subscription/', include('subscription.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
