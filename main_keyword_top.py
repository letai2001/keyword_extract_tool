import json
from collections import defaultdict
from datetime import datetime
from langdetect import detect 
import numpy as np
import string
import re
import matplotlib.pyplot as plt


# Tạo cấu trúc dữ liệu cho thống kê
def stat_keyword(start_datestr , end_datestr  , data):
    keyword_stats = defaultdict(lambda: defaultdict(int))
    # Thực hiện thống kê

    date_counts = {}

    # Định dạng ngày và khoảng thời gian cần xem xét
    date_format = "%m/%d/%Y"
    start_date = datetime.strptime(start_datestr, date_format)
    end_date = datetime.strptime(end_datestr, date_format)

    # Tạo từ điển để theo dõi số lần xuất hiện của mỗi ngày và từ điển cho keyword
    date_counts = defaultdict(int)
    keyword_counts = defaultdict(lambda: defaultdict(int))

    # Duyệt qua từng mục trong dữ liệu
    for item_id, item in data.items():
        # if(is_valid_vietnamese_text(item['content'] , 15)):
            date_str = item['created_time']
            date_str = datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S').strftime('%m/%d/%Y')
            date = datetime.strptime(date_str, date_format)

            # Kiểm tra nếu ngày nằm trong khoảng thời gian đã định
            if start_date <= date <= end_date:
                date_counts[date_str] += 1
                for keyword in item['keywords']:
                    keyword_counts[date_str][keyword] += 1

    # Tính phần trăm xuất hiện của từng keyword
    keyword_percentages = defaultdict(dict)
    for date, keywords in keyword_counts.items():
        for keyword, count in keywords.items():
            if date_counts[date] > 0:
                keyword_percentages[date][keyword] = (count / date_counts[date]) * 100
    output_data = []

    def is_keyword_selected(keyword, keyword_percentages, check_date_str):
        
        percentage_on_check_date = keyword_percentages[check_date_str].get(keyword, 0)
        
        # Lấy phần trăm của từ trong các ngày khác
        other_dates_percentages = [keyword_percentages[date][keyword] 
                                for date in keyword_percentages 
                                #    if date != check_date_str and keyword in keyword_percentages[date]
                                        if  keyword in keyword_percentages[date]
                                ]
        
        # Đếm số ngày có phần trăm lớn hơn 0.6% và 0.8%
        #trung bình số phần trăm của từ  top 20 mỗi ngày 
        count_higher_09 = sum(perc >= 0.88 for perc in other_dates_percentages)
        #trung bình số phần trăm của từ  top 10 mỗi ngày 
        count_higher_11 = sum(perc >= 1.1 for perc in other_dates_percentages)
        #trung bình số phần trăm của từ  top 6 mỗi ngày 
        count_higher_14 = sum(perc >= 1.4 for perc in other_dates_percentages)

        # Tiêu chí 1: Từ không được chọn nếu có tới 4 ngày lớn hơn 0.6% và 3 ngày lớn hơn 0.8%
        if count_higher_09 >= 4 and count_higher_11 >= 2:
            # Tiêu chí 2: Từ được chọn nếu phần trăm trong ngày check_date gấp 2.75 lần trung bình các ngày còn lại        
            # avg_other_dates = sum(other_dates_percentages) / len(other_dates_percentages) if other_dates_percentages else 0
            min_other_percentage = min(other_dates_percentages) if other_dates_percentages else 0
            if percentage_on_check_date > 3 * min_other_percentage and count_higher_14 <=5:
                return True
            else:
                return False
        else:
            # Từ được chọn theo tiêu chí 1
            return True   
    for date, keywords in keyword_percentages.items():
        # Sắp xếp các keyword dựa trên phần trăm xuất hiện
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        
        # Tạo danh sách các từ và phần trăm tương ứng
        keywords_data = [{"keyword": keyword, "percentage": percentage} for keyword, percentage in sorted_keywords]
        keywords_data_top =  [{"keyword": keyword, "percentage": percentage} for keyword, percentage in sorted_keywords if is_keyword_selected(keyword, keyword_percentages, date)]
        # Thêm vào danh sách output
        output_data.append({"date": date,  "keywords_top": keywords_data_top , "keywords": keywords_data })

    # Lưu vào file JSON
    with open('keyword_percentages_main_title.json', 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)
        return output_data
    
    
if __name__ == '__main__':
    with open('keyword_test_27.1_filter_new.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    start_str = "12/07/2023"
    end_str = "12/13/2023"
    stat_keyword(start_str , end_str , data)