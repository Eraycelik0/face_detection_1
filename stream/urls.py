from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),  # Ana sayfa
    path('upload_face_image/', views.upload_face_image, name='upload_face_image'),
    path('video_feed/', views.video_feed, name='video_feed'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
