# functions for preprocessing various fields of the raw data
import collections
import scipy.sparse as sp

__all__ = ["taglist_to_matrix",
           ]


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
