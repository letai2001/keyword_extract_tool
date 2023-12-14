import numpy as np
import re
from vncorenlp import VnCoreNLP
import json
from langdetect import detect 
from langdetect import detect as detect2
from main_query_es import query_day
from main_keyword_extract import extract_keyword_title
from datetime import datetime, timedelta
from collections import defaultdict

with open('vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
    stop_words = f.read().splitlines()
    

start_date = "2023-12-07"
end_date = "2023-12-14"
start_date = datetime.strptime(start_date, "%Y-%m-%d")
end_date = datetime.strptime(end_date, "%Y-%m-%d")
start_date_str = start_date.strftime("%Y/%m/%d 00:00:01")
end_date_str = end_date.strftime("%Y/%m/%d 00:00:00")
dataFramse_Log = query_day(start_date_str , end_date_str)
vn_core = VnCoreNLP("C:\\Users\\Admin\\Downloads\\vncorenlp\\VnCoreNLP\\VnCoreNLP-1.2.jar" ,  annotators="wseg,pos", max_heap_size='-Xmx2g')

data_title_dict = extract_keyword_title(dataFramse_Log , vn_core , stop_words)
