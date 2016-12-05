import os
import urllib.request


__all__ = ["download"]


def download_files(bucket_path, file_list, download_path):
    """
    Provide path to s3 bucket, download a file list to download path
    """
    if not os.path.isdir(download_path):
        os.makedirs(download_path)
    for f in file_list:
        # check if file already exists
        file_path = os.path.join(download_path, f)
        if os.path.isfile(file_path):
            print ('File "%s" already exists' %f)
        else:
            print ('Downloading "%s" ...' % f)
            urllib.request.urlretrieve(bucket_path + f, file_path)
            print ('Done')


def download(file_list=[]):
    """
    Downloads files from AWS S3 repository
    Here are all avialble dataset from the repository
    file_list=["yelp_academic_dataset_business.pickle"
               "yelp_academic_dataset_review.pickle",
               "yelp_academic_dataset_user.pickle",
               "yelp_academic_dataset_checkin.pickle",
               "yelp_academic_dataset_tip.pickle"]
    """
    if file_list == []:
        print ("Providiing empty file_list, no download...")
    else:
        bucket_path = "https://s3-us-west-2.amazonaws.com/science-of-science-bucket/yelp_academic_dataset/"
        current_path = os.path.dirname(os.path.abspath(__file__))
        download_path = os.path.join(current_path, '..', 'data')
        download_files(bucket_path, file_list, download_path)
