import json
from collections import defaultdict
from datetime import datetime
from langdetect import detect 
import numpy as np
import string
import re

def is_valid_vietnamese_text(text, word_limit=15):
    try:
        # Chỉ xét một số từ đầu tiên
        words = text.split()[:word_limit]
        sample_text = ' '.join(words)

        # Kiểm tra ngôn ngữ của đoạn văn bản mẫu
        if detect(sample_text) != 'vi':
            return False

        # Sử dụng biểu thức chính quy để tìm các ký tự không phải là chữ cái
        non_alpha_pattern = re.compile(r'[^a-zA-ZàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ\s]')
        non_alpha_chars = non_alpha_pattern.findall(sample_text)

        if len(non_alpha_chars) / len(sample_text) > 0.1: # Thay đổi tỷ lệ này tùy theo nhu cầu
            return False

        return True
    except:
        return False

# Đọc dữ liệu từ file JSON
with open('update_keyword_19.1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
import matplotlib.pyplot as plt

# Tạo cấu trúc dữ liệu cho thống kê
keyword_stats = defaultdict(lambda: defaultdict(int))
# Thực hiện thống kê

date_counts = {}

# Định dạng ngày và khoảng thời gian cần xem xét
date_format = "%m/%d/%Y"
start_date = datetime.strptime("11/19/2023", date_format)
end_date = datetime.strptime("11/26/2023", date_format)

# Tạo từ điển để theo dõi số lần xuất hiện của mỗi ngày và từ điển cho keyword
date_counts = defaultdict(int)
keyword_counts = defaultdict(lambda: defaultdict(int))

# Duyệt qua từng mục trong dữ liệu
for item_id, item in data.items():
    # if(is_valid_vietnamese_text(item['content'] , 15)):
        date_str = item['created_time']
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
check_date_str = "11/26/2023"
check_date =  datetime.strptime("11/26/2023", date_format)
# for date, keywords in keyword_percentages.items():
#     # Sắp xếp các keyword dựa trên phần trăm xuất hiện
#     sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
sorted_keywords = sorted(keyword_percentages[check_date_str].items(), key=lambda x: x[1], reverse=True)

def is_keyword_selected(keyword, keyword_percentages, check_date_str, date_counts):
    """
    Kiểm tra xem một từ khóa có được chọn hay không dựa trên các tiêu chí đề ra.

    Args:
    keyword (str): Từ khóa cần kiểm tra.
    keyword_percentages (defaultdict): Từ điển chứa phần trăm xuất hiện của từ khóa theo ngày.
    check_date_str (str): Ngày cần kiểm tra dạng chuỗi.
    date_counts (defaultdict): Từ điển chứa số lượng lần xuất hiện của mỗi ngày.

    Returns:
    bool: Trả về True nếu từ khóa được chọn, ngược lại là False.
    """
    
    # Lấy phần trăm của từ trong ngày check_date
    percentage_on_check_date = keyword_percentages[check_date_str].get(keyword, 0)
    
    # Lấy phần trăm của từ trong các ngày khác
    other_dates_percentages = [keyword_percentages[date][keyword] 
                               for date in keyword_percentages 
                               if date != check_date_str and keyword in keyword_percentages[date]]
    
    # Đếm số ngày có phần trăm lớn hơn 0.6% và 0.8%
    count_higher_06 = sum(perc > 0.6 for perc in percentage_on_check_date+other_dates_percentages)
    count_higher_08 = sum(perc > 0.8 for perc in percentage_on_check_date+other_dates_percentages)

    # Tiêu chí 1: Từ không được chọn nếu có tới 4 ngày lớn hơn 0.6% và 3 ngày lớn hơn 0.8%
    if count_higher_06 >= 4 and count_higher_08 >= 3:
        # Tiêu chí 2: Từ được chọn nếu phần trăm trong ngày check_date gấp 2.75 lần trung bình các ngày còn lại
        avg_other_dates = sum(other_dates_percentages) / len(other_dates_percentages) if other_dates_percentages else 0
        if percentage_on_check_date > 2.75 * avg_other_dates:
            return True
        else:
            return False
    else:
        # Từ được chọn theo tiêu chí 1
        return True


# Lưu kết quả vào file
with open('keyword_percentages_19.1_title.txt', 'w' , encoding='utf-8') as file:
    for date, keywords in keyword_percentages.items():
        # Sắp xếp các keyword dựa trên phần trăm xuất hiện
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        file.write(f"Ngày {date}:\n")
        for keyword, percentage in sorted_keywords:
            file.write(f"   Keyword '{keyword}': {percentage:.2f}%\n")
keyword_to_analyze = "trung_quốc"
with open('selected_keywords.txt', 'w', encoding='utf-8') as file:
    for date, keywords in keyword_percentages.items():
        # Sắp xếp các keyword dựa trên phần trăm xuất hiện
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        
        # Lọc và ghi ra các từ khóa được chọn
        file.write(f"Ngày {date}:\n")
        for keyword, _ in sorted_keywords:
            if is_keyword_selected(keyword, keyword_percentages, date, date_counts):
                percentage = keyword_percentages[date][keyword]
                file.write(f"   Keyword '{keyword}': {percentage:.2f}%\n")

# Chuẩn bị dữ liệu
dates = sorted(keyword_percentages.keys(), key=lambda date: datetime.strptime(date, "%m/%d/%Y"))
percentages = [keyword_percentages[date].get(keyword_to_analyze, 0) for date in dates]

# Chuyển đổi chuỗi ngày thành đối tượng datetime để vẽ biểu đồ
dates = [datetime.strptime(date, "%m/%d/%Y") for date in dates]


# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
plt.plot(dates, percentages, marker='o')
plt.title(f"Biến thiên phần trăm của từ '{keyword_to_analyze}' theo ngày")

# Thêm chú thích số lần xuất hiện của từ và số bài trong mỗi ngày
for i, date in enumerate(dates):
    date_str = date.strftime('%m/%d/%Y')
    # Chú thích số lần xuất hiện của từ
    plt.annotate(f"{keyword_counts[date_str].get(keyword_to_analyze, 0)} lần",
                 (date, percentages[i]),
                 textcoords="offset points",  # Vị trí của chú thích
                 xytext=(0,10),  # Khoảng cách từ điểm đến chú thích
                 ha='center')  # Căn giữa chú thích

    # Chú thích số bài trong ngày trên trục hoành
    plt.annotate(f"{date_counts[date_str]} bài",
                 (date, 0),
                 textcoords="offset points",  # Vị trí của chú thích
                 xytext=(0, -20),  # Khoảng cách từ điểm đến chú thích
                 ha='center',  # Căn giữa chú thích
                 color='red')  # Màu sắc chú thích

plt.xlabel("Ngày")
plt.ylabel("Phần trăm xuất hiện")
plt.xticks(dates, rotation=45)
plt.grid(True)
plt.show()


