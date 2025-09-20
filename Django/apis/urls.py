from django.urls import path
from apis import views

urlpatterns = [
    path('', views.index),
]
