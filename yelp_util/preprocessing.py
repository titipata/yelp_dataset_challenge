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
