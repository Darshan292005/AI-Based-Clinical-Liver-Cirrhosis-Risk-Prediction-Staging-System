from django.urls import path
from . import views

urlpatterns = [
    path('ask_chatbot', views.ask_chatbot, name='ask_chatbot'),
]
