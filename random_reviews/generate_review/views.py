from django.shortcuts import render_to_response
from django.http import HttpResponse
from django.http import Http404 # 404
from django.template import RequestContext
from django.http import JsonResponse


def index(request):
    return render_to_response('index.html')

import os
from numpy import random
import numpy as np
import pandas as pd
from sklearn.externals import joblib

try:
    print "Loading first sentences..."
    first_sentence_features = pd.read_pickle('/mnt/aida/daniel/data/first_sentence_features.pickle')
    path = '/home/ubuntu/regression_model/'
    lregressions = []
    for i in range(100):
        model_name = 'regression_%s.pkl'% i
        lregressions.append(joblib.load(os.path.join(path, model_name)))
    print "Loading review sentences..."
    review_sentences_df = pd.read_pickle('/mnt/aida/daniel/data/review_sentences_df.pickle')
    print "Loading nearest neighbor model..."
    nrnb = joblib.load('/mnt/aida/daniel/data/nearest_neighbor_yelp.pkl')
    first_sentence_features = first_sentence_features[first_sentence_features.first_sentence.map(len) <= 100]
    first_sentence_features.reset_index(inplace=True, drop=True)
    local = False
except:
    print "Not loeding any of the model"
    local = True # local testing


# function for generating sentences
def generate_sentences(business_type, stars, cool, funny, useful, n_sentences = 10):
    review_sentences = []
    # generate first sentence
    all_first_sentences = first_sentence_features.query('stars == %u' % stars).query('cluster == %u' % business_type)
    fs = all_first_sentences.iloc[random.randint(0, all_first_sentences.shape[0]-1)]
    review_sentences.append(fs.first_sentence)
    # features
    x = [np.hstack((fs.wordvect, fs[0:5].tolist()))]
    y = np.hstack([lr.predict(x) for lr in lregressions])
    distance, index = nrnb.kneighbors(X = [y])

    next_idx = index[0][random.choice(len(index[0]), p = (1/distance[0])/((1/distance[0]).sum()))]

    for _ in range(n_sentences):
        current_wordvect = review_sentences_df.iloc[next_idx].wordvect
        review_sentences.append(review_sentences_df.iloc[next_idx].sentence)

        x = [np.hstack((current_wordvect, [stars, cool, funny, useful, business_type]))]
        y = np.hstack([lr.predict(x) for lr in lregressions])
        distance, index = nrnb.kneighbors(X = [y])
        next_idx = index[0][random.choice(len(index[0]), p = (1/distance[0])/((1/distance[0]).sum()))]

    return review_sentences

def generate_review(request, business_type, stars, funny, cool, useful):
    if local is not True:
        review_sentences = generate_sentences(int(business_type), int(stars), int(cool), int(funny), int(useful), n_sentences=5)
        review_list = ['^500 '.join(review_sentences)]
    else:
        review_list = ['Nothing but this input ' + str(business_type) + ' ' + str(stars) + ' ' + str(funny) + ' ' + str(cool) + ' ' + str(useful)]
    return JsonResponse({'review': review_list})
