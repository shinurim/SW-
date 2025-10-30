from django.urls import path
from apis import views

urlpatterns = [
    path("api/get-data", views.get_data, name="get_data"),
]