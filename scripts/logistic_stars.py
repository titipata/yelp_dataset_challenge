# using reviews to predict stars

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import WhitespaceTokenizer

w_tokenizer = WhitespaceTokenizer()

def clean_string(string):
    """
    Clean string
    """
    clean_string = re.sub('\.', ' ', string)
    clean_string = re.sub('\*', '', clean_string)
    clean_string =  re.sub('-', ' ', clean_string)
    clean_string =  re.sub(',', ' ', clean_string)
    clean_string =  re.sub("'", ' ', clean_string)
    clean_string =  re.sub('"', ' ', clean_string)
    clean_string = ' '.join(w_tokenizer.tokenize(clean_string)).strip()
    return clean_string

# ref: https://www.kaggle.com/c/crowdflower-search-relevance
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


# simple logistic regression on review
def run():
    review = pd.read_pickle('../data/yelp_academic_dataset_review.pickle')
    clean_review = review.text.map(clean_string)
    review_stars = pd.concat((clean_review, review.stars), axis=1) # text and stars
    review_train, review_test, y_train, y_test = cross_validation.train_test_split(review_stars.text, review_stars.stars,
                                                                               test_size=0.2, random_state=0)


    tfidf_model = TfidfVectorizer(min_df=2, max_df=0.8, strip_accents='unicode',
                                  analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2),
                                  use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words='english')
    logistic_model = LogisticRegression(penalty='l2', C=1.0)
    X_train = tfidf_model.fit_transform(list(review_train[0::]))
    X_test = tfidf_model.transform(list(review_test[0::]))
    logistic_model.fit(X_train, y_train)
    y_hat = logistic_model.predict(X_test)
    print "Quadratic weight kappa = ", quadratic_weighted_kappa(y_test, y_hat)
