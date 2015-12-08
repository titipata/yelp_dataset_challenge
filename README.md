# Yelp Dataset Challenge for Python

Repository for reading and downloading [Yelp Dataset Challenge](http://www.yelp.com/dataset_challenge)
round 6 in Pandas pickle format. This repository makes it easy for anyone who want to mess around with Yelp data using Python.
I provide `yelp_util` Python package that has read and download function.

## Datasets repository

The following is structure of S3,

```bash
science-of-science-bucket
└─yelp_academic_dataset
  ├───yelp_academic_dataset_business.pickle (61k rows)
  ├───yelp_academic_dataset_review.pickle (1.5M rows)
  ├───yelp_academic_dataset_user.pickle (366k rows)
  ├───yelp_academic_dataset_checkin.pickle (45k rows)
  └───yelp_academic_dataset_tip.pickle (495k rows)
```

You can download data directly from AWS S3 repository as follows,

```python
import yelp_util
yelp_util.download(file_list=["yelp_academic_dataset_business.pickle",
                              "yelp_academic_dataset_review.pickle",
                              "yelp_academic_dataset_user.pickle",
                              "yelp_academic_dataset_checkin.pickle",
                              "yelp_academic_dataset_tip.pickle"])
```

The file will be downloaded to `data` folder. After finishing download, you can simply read
`pickle` as follows

```python
import pandas as pd
review = pd.read_pickle('data/yelp_academic_dataset_review.pickle')
review.head()
```


## Structure of Datasets

**User** table of user's information (366k rows)

average_stars | compliments | elite | fans | friends | name | review_count | type | user_id | votes | yelping_since
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |


**Business** table of business with its location and city that it locates (61k rows)

attributes | business_id | categories	| city | full_address | hours | latitude | longitude | name | neighborhoods | open | review_count | stars | state | type
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

**Review** reviews made by users (1.5M rows)

business_id | date | review_id | stars | text | type | user_id | type | votes_cool | votes_funny | votes_useful
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

**Checkin** check-in table (45k rows)

business_id | checkin_info | type |
:---: | :---: | :---: |

**Tip** tip table (495k rows)

business_id | date | likes | text | type | user_id |
---: | :---: | :---: | :---: | :---: |  :---: |


## Members

- [Titipat Achakulvisut](http://titipata.github.io)
- [Daniel Acuna](http://www.scienceofscience.org)
