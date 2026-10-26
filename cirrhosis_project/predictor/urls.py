from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict', views.predict, name='predict'),
    path('result/', views.result_page, name='result_page'),
]
