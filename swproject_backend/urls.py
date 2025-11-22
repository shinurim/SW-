# swproject_backend/urls.py
from django.urls import path, include

urlpatterns = [
    path("api/v1/", include("rdb.urls")),   # ★ rdb에 /api/v1/ 프리픽스 붙여서 사용
    path("api/v1/insight/", include("insight.urls")),
    path("api/v1/", include("apis.urls")),   #/api/v1/save/save_segment , save_user
]
