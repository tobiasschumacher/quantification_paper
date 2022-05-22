import pandas as pd


def prep_data(binned=False):
    colnames = ['word_freq_make',
                'word_freq_address',
                'word_freq_all',
                'word_freq_3d',
                'word_freq_our',
                'word_freq_over',
                'word_freq_remove',
                'word_freq_internet',
                'word_freq_order',
                'word_freq_mail',
                'word_freq_receive',
                'word_freq_will',
                'word_freq_people',
                'word_freq_report',
                'word_freq_addresses',
                'word_freq_free',
                'word_freq_business',
                'word_freq_email',
                'word_freq_you',
                'word_freq_credit',
                'word_freq_your',
                'word_freq_font',
                'word_freq_000',
                'word_freq_money',
                'word_freq_hp',
                'word_freq_hpl',
                'word_freq_george',
                'word_freq_650',
                'word_freq_lab',
                'word_freq_labs',
                'word_freq_telnet',
                'word_freq_857',
                'word_freq_data',
                'word_freq_415',
                'word_freq_85',
                'word_freq_technology',
                'word_freq_1999',
                'word_freq_parts',
                'word_freq_pm',
                'word_freq_direct',
                'word_freq_cs',
                'word_freq_meeting',
                'word_freq_original',
                'word_freq_project',
                'word_freq_re',
                'word_freq_edu',
                'word_freq_table',
                'word_freq_conference',
                'char_freq_;',
                'char_freq_(',
                'char_freq_[',
                'char_freq_!',
                'char_freq_$',
                'char_freq_#',
                'capital_run_length_average',
                'capital_run_length_longest',
                'capital_run_length_total',
                'spam']

    # url = ["https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"]

    dta = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
                      names=colnames,
                      index_col=False,
                      skipinitialspace=True)

    # dta.to_pickle("dta.pkl")

    if binned:
        for col in list(dta)[0:-1]:
            dta[col] = pd.cut(dta[col], bins=5, labels=[1, 2, 3, 4, 5])
            dta[col] = dta[col].astype("int64")
            # dta.to_pickle("dta_binned.pkl")

    return dta
