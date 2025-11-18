# Django_proj/insight/urls.py
from django.urls import path
from .views_insight import generate_insight

urlpatterns = [
    path("from-text", generate_insight),   # POST /api/v1/insight/from-text
]
