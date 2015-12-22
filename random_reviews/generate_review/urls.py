from django.conf.urls import patterns, include, url
from generate_review import views

urlpatterns = patterns('',
    url(r'^$', views.index, name='index'),
)
