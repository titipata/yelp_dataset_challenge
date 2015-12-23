from django.conf.urls import patterns, include, url
from generate_review import views

urlpatterns = patterns('',
    url(r'^$', views.index, name='index'),
    url(r'^generate_review/(?P<business_type>[0-3])/(?P<stars>[1-5])/(?P<funny>[0-9]+)/(?P<cool>[0-9]+)/(?P<useful>[0-9]+)$', views.generate_review, name='generate_review'),
)
