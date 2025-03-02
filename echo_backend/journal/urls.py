from django.urls import path
from . import views

urlpatterns = [
    path('journals/', views.journal_view, name='journal_view'),  # Handles GET & POST
    path('journals/<int:journal_id>/', views.journal_detail_view, name='journal_detail_view'),  # Handles PUT & DELETE
]
