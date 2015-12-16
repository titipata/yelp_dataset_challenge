# functions for preprocessing various fields of the raw data
import re
import time
import collections
import scipy.sparse as sp
import nltk.data
import tensorflow as tf
from nltk.tokenize import WhitespaceTokenizer
from gensim.models import Word2Vec
from unidecode import unidecode
from itertools import chain
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer


__all__ = ["taglist_to_matrix",
           "create_word2vec_model",
           "clear_tensorflow_graph",
           "get_stream_seq",
           "get_word_embedding"
           ]


sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
whitespace_tokenizer = WhitespaceTokenizer()


def taglist_to_matrix(taglist):
    """
    This function
    Args:
        taglist: list of list of tags. For example, each element of the list is the list of tags of a business category:
        [u'Doctors', u'Health & Medical']

    Returns:
        A sparse matrix num_docs x tags where element i, j has the counts of how many time tag j appear in document i
    """

    all_tags = [w for doc in taglist for w in doc]
    counter = collections.Counter(all_tags)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    # sparse matrix indices
    i_indices = [doc_idx for doc_idx in range(len(taglist)) for _ in taglist[doc_idx]]
    j_indices = [word_to_id[w] for doc_idx in range(len(taglist)) for w in taglist[doc_idx]]
    data = [1]*len(all_tags)
    m = sp.csc_matrix((data, (i_indices, j_indices)))
    m.sum_duplicates()
    return m


def clean_text(text):
    """Clean and lower string
    Parameters
    ----------
        text : in string format
    Returns
    -------
        text_clean : clean text input in string format
    """
    text_clean = re.sub(':', '', text.lower())
    text_clean = re.sub(',', '', text_clean)
    text_clean = re.sub('\.', '', text_clean)
    return text_clean


def clean_and_tokenize(text):
    """
    Divide review into sentence, clean words,
    and tokenize it.

    Returns
    ------
        text_tokenize: list of word in sentence
    """
    sentence = sent_detector.tokenize(unidecode(text))
    text_clean = map(clean_text, sentence)
    text_tokenize = map(lambda x: whitespace_tokenizer.tokenize(x), text_clean)
    return text_tokenize


def clean_and_tokenize_word(text):
    """
    Clean and divide text (review) into list of words
    Returns
    ------
        text_clean: list of word in sentence
    """
    if isinstance(text, list):
        text_clean = map(clean_text, text)
        text_tokenize = map(whitespace_tokenizer.tokenize, text_clean)
    elif isinstance(text, basestring):
        text_clean = clean_text(text)
        text_tokenize = whitespace_tokenizer.tokenize(text_clean)
    else:
        text_tokenize = []
    return text_tokenize


def create_word2vec_model(review_list, size=100, window=5, min_count=10, workers=16):
    """
    Create gensim Word2Vec model out of review list
    where each element contains review
    """
    print 'breaking into sentence...'
    review_sentence = map(clean_and_tokenize, review_list)
    review_flatten = list(chain.from_iterable(review_sentence))
    print 'trianing word2vec model...'
    word2vec_model = Word2Vec(review_flatten, size=size, window=window, min_count=min_count, workers=16)
    return word2vec_model


def clear_tensorflow_graph():
    """
    Clear all Tensor graph
    """
    tf.ops.reset_default_graph()


def get_stream_seq(review_list, word2vec_model):
    """
    From review list and word2vec model,
    generate output stream of output of review index
    correspond to concatenated review list
    """
    review_list_clean = clean_and_tokenize_word(review_list)
    review_list_flatten = list(chain.from_iterable(review_list_clean))
    review_words_stream = filter(lambda x: x is not None,
                             map(lambda x: word2vec_model.vocab.get(x).index if x in word2vec_model.vocab else None,
                                 review_list_flatten)
                                 )
    return review_words_stream


def get_word_embedding(word2vec_model):
    embeddings = word2vec_model.syn0
    print 'Vocabulary size: ', embeddings.shape[0]
    print 'Word vector dimension: ', embeddings.shape[1]
    return embeddings

def create_vocab(review_list):
    """
    Create dictionary out of review list
    ref: http://deeplearning.net/tutorial/lstm.html
    """

    tb_tokenizer = TreebankWordTokenizer()
    # Tokenized sentences
    tksents = [tb_tokenizer.tokenize(review) for review in review_list]
    print('Building dictionary..')
    wordcount = dict()
    for sent in tksents:
        for w in sent:
            if w.lower() not in wordcount:
                wordcount[w.lower()] = 1
            else:
                wordcount[w.lower()] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = np.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print(np.sum(counts), ' total words ', len(keys), ' unique words')

    return worddict, tksents

def word2id(tksents, dictionary):

    seqs = [None] * len(tksents)
    for idx, ss in enumerate(tksents):
        seqs[idx] = [dictionary[w.lower()] if w.lower() \
                        in dictionary else 1 for w in ss]

    return seqs

def load_yelp_review(X, labels, nb_words=None, skip_top=10,\
                        maxlen=None, test_split=0.2, seed=113, oov_char=1):
    '''
        Preprocess and load Yelp Reviews word2id sequences and labels for
        polarity analysis

        nb_words : Maximum number of words to index, else assign oov_char
        skip_top : Skip n top most common words
        maxlen   : Maximum sequence length
        oov_char : Out-Of-Vocabulary word id
        test_split : Train-Test split

        ref:https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
    '''

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels

    if not nb_words:
        nb_words = max([max(x) for x in X])

    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X_train = X[:int(len(X)*(1-test_split))]
    y_train = labels[:int(len(X)*(1-test_split))]

    X_test = X[int(len(X)*(1-test_split)):]
    y_test = labels[int(len(X)*(1-test_split)):]

    return (X_train, y_train), (X_test, y_test)
