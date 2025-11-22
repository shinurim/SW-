from django.urls import path
from .views_panel import rdb_gateway

urlpatterns = [
    path("search/text", rdb_gateway, name="search_text"),
    path("search/sql",  rdb_gateway, name="search_sql"),
]
