from django.urls import path
from api import views

urlpatterns = [
    path('',views.index_page),
    path('result', views.yolov4tiny_outcome),
]