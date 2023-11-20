import json
from datetime import datetime

# Đọc dữ liệu từ file media.json
with open('content_test_new.json', 'r' , encoding='utf-8') as file:
    media_data = json.load(file)

# Đọc dữ liệu từ file keyword.json
with open('keyword_test_3.json', 'r' , encoding='utf-8') as file:
    keyword_data = json.load(file)

# Hợp nhất dữ liệu
for item in media_data:
    _id = item['_id']
    _type = item['_source']['type']
    time_crawl = item['_source']['time_crawl']
    # Chỉ lấy ngày từ time_crawl
    date_only = datetime.strptime(time_crawl, '%m/%d/%Y %H:%M:%S').strftime('%m/%d/%Y')

    if _id in keyword_data:
        keyword_data[_id]['_type'] = _type
        keyword_data[_id]['time_crawl'] = date_only

# Lưu dữ liệu đã hợp nhất
with open('update_keyword.json', 'w' , encoding='utf-8') as file:
    json.dump(keyword_data, file, ensure_ascii=False ,indent=4)
