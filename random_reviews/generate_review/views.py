from django.shortcuts import render_to_response
from django.http import HttpResponse
from django.http import Http404 # 404
from django.template import RequestContext


def index(request):
    return render_to_response('index.html')
