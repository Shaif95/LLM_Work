from django.urls import path
from .views import upload_pdf, answer

urlpatterns = [
    path('upload/', upload_pdf, name='upload_pdf'),
    path('answer/', answer, name="answer")
]
