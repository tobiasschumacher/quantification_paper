# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 19:29:58 2018

@author: tobi_
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 22:39:29 2018

@author: tobi_
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:53:42 2018

@author: tobi_
"""

import pandas as pd

def prep_data(binned = False):

  f = "data/phishing/phishing.csv"
  
  colnames = ['having_IP_Address',
        'URL_Length',
        'Shortining_Service',
        'having_At_Symbol',
        'double_slash_redirecting',
        'Prefix_Suffix',
        'having_Sub_Domain',
        'SSLfinal_State',
        'Domain_registeration_length',
        'Favicon',
        'port',
        'HTTPS_token',
        'Request_URL',
        'URL_of_Anchor',
        'Links_in_tags',
        'SFH',
        'Submitting_to_email',
        'Abnormal_URL',
        'Redirect',
        'on_mouseover',
        'RightClick',
        'popUpWidnow',
        'Iframe',
        'age_of_domain',
        'DNSRecord',
        'web_traffic',
        'Page_Rank',
        'Google_Index',
        'Links_pointing_to_page',
        'Statistical_report',
        'Result']
  
  dta = pd.read_csv(f,
              names = colnames,
              index_col = False,
              skipinitialspace = True)
  
  return dta

#dta.to_pickle("dta.pkl")
