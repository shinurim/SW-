from django.urls import path
from .views_save import save_segment,retrieve_segment,list_segments,save_user


urlpatterns = [
    path("save/save_segment", save_segment),   # POST /api/v1/save/save_segment
    path("save/save_user", save_user),   # POST /api/v1/save/save_user
    path("segments", list_segments),   # GET /api/v1/segments?user_id=xxx
    path("insights/<int:segment_id>", retrieve_segment),   # GET /api/v1/insights/<segment_id>?user_id=xxx
]