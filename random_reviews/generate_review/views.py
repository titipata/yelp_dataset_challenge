from django.shortcuts import render_to_response
from django.http import HttpResponse
from django.http import Http404 # 404
from django.template import RequestContext
from django.http import JsonResponse


def index(request):
    return render_to_response('index.html')


def generate_review(request, business_type, stars, funny, cool, useful):
    return JsonResponse({'review': [
                        business_type + ' ' + stars + ' ' + funny + ' ' + cool + ' ' + useful,
                        'This business is great.',
                        'I will come back soon']})
