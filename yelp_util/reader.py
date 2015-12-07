# reader function for Yelp to Pandas DataFrame format
import os
import json
import pandas as pd
from glob import glob

__all__ = ["list_json",
           "read_yelp_json",
           "read_yelp_review"]


def list_json(path):
    """
    Provide path to json folder
    list all path
    """
    path_to_json = os.path.join(path, '*.json')
    return glob(path_to_json)


def convert(x):
    ''' Convert a json string to a flat python dictionary
    which can be passed into Pandas.

    reference: https://gist.github.com/paulgb/5265767
    '''
    ob = json.loads(x)
    for k, v in ob.items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob


def read_yelp_json(path_to_json):
    """
    Read json file and return as Pandas DataFrame
    Example:
        df_review = read_yelp_json('yelp_academic_dataset_review.json')

    reference: https://www.reddit.com/r/MachineLearning/comments/33eglq/python_help_jsoncsv_pandas/
    """
    with open('/home/ubuntu/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json', 'rb') as f:
        data = f.readlines()
    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ','.join(data) + "]"
    df = pd.read_json(data_json_str)
    return df


def read_yelp_review(path_to_json):
    """
    Read Yelp review data and return as Pandas DataFrame
    reference: https://gist.github.com/paulgb/5265767
    """
    df = pd.DataFrame([convert(line) for line in file(path_to_json)])
    return df
