from django.urls import path
from .views import home, search, detail

urlpatterns = [
    path('', home, name='home'),
    path('search/', search, name='search'),
    path('detail/<str:doc_id>', detail, name='detail')
]