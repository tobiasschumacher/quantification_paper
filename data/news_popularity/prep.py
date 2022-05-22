from io import BytesIO
from zipfile import ZipFile
import urllib.request
import pandas as pd


def prep_data(binned=False):
    url = urllib.request.urlopen(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip")

    my_zip_file = ZipFile(BytesIO(url.read()))
    dta_file = my_zip_file.namelist()[2]

    dta = pd.read_csv(my_zip_file.open(dta_file),
                      skipinitialspace=True)

    dta = dta.drop(columns="url")

    bins = [0, 1000, 2500, 5000, 1000000]
    labels = [1, 2, 3, 4]
    dta['shares'] = pd.cut(dta['shares'], bins=bins, labels=labels)
    dta['shares'] = dta['shares'].astype("int64")

    # dta.to_pickle("dta.pkl")

    if binned:
        qcols = ['timedelta',
                 'n_tokens_title',
                 'n_tokens_content',
                 'n_unique_tokens',
                 'n_non_stop_words',
                 'n_non_stop_unique_tokens',
                 'num_hrefs',
                 'num_self_hrefs',
                 'num_imgs',
                 'num_videos',
                 'average_token_length',
                 'num_keywords',
                 'kw_min_min',
                 'kw_max_min',
                 'kw_avg_min',
                 'kw_min_max',
                 'kw_max_max',
                 'kw_avg_max',
                 'kw_min_avg',
                 'kw_max_avg',
                 'kw_avg_avg',
                 'self_reference_min_shares',
                 'self_reference_max_shares',
                 'self_reference_avg_sharess',
                 'LDA_00',
                 'LDA_01',
                 'LDA_02',
                 'LDA_03',
                 'LDA_04',
                 'global_subjectivity',
                 'global_sentiment_polarity',
                 'global_rate_positive_words',
                 'global_rate_negative_words',
                 'rate_positive_words',
                 'rate_negative_words',
                 'avg_positive_polarity',
                 'min_positive_polarity',
                 'max_positive_polarity',
                 'avg_negative_polarity',
                 'min_negative_polarity',
                 'max_negative_polarity',
                 'title_subjectivity',
                 'title_sentiment_polarity',
                 'abs_title_subjectivity',
                 'abs_title_sentiment_polarity']
        for col in qcols:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

    return dta

# dta.to_pickle("dta_binned.pkl")
