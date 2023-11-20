import json
from collections import defaultdict

# Đọc dữ liệu từ file JSON
with open('update_keyword.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Tạo cấu trúc dữ liệu cho thống kê
keyword_stats = defaultdict(lambda: defaultdict(int))
base_date = "11/10/2023"  # Ngày gốc
base_date_percentages = defaultdict(float)
# Thực hiện thống kê
for item_id, item in data.items():
    if item['_type'] == "electronic media":
        date = item['time_crawl']
        for keyword in item['keywords']:
            keyword_stats[date][keyword] += 1

# Tính phần trăm
keyword_percentages = defaultdict(dict)
for date, keywords in keyword_stats.items():
    total_keywords = sum(keywords.values())
    for keyword, count in keywords.items():
        percentage = (count / total_keywords) * 100
        keyword_percentages[date][keyword] = percentage
        if date == base_date:
            base_date_percentages[keyword] = percentage
with open('keyword_percentages_2.txt', 'w', encoding='utf-8') as file:
    for date, keywords in sorted(keyword_stats.items()):
        total_keywords = sum(keywords.values())
        percentages = {keyword: (count / total_keywords) * 100 for keyword, count in keywords.items()}
        # Sắp xếp từ khóa theo phần trăm xuất hiện
        sorted_percentages = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

        file.write(f"Ngày {date}:\n")
        for keyword, percentage in sorted_percentages:
            file.write(f"- {keyword}: {percentage:.2f}%\n")
        file.write("\n")

# Tính và xuất sự thay đổi phần trăm so với ngày gốc
with open('keyword_percentage_changes_2.txt', 'w', encoding='utf-8') as file:
    for date, keywords in sorted(keyword_percentages.items()):
        if date != base_date:
            changes = {}
            for keyword, percentage in keywords.items():
                base_percentage = base_date_percentages.get(keyword, 0)
                change = percentage - base_percentage  # Sử dụng giá trị thực
                changes[keyword] = change
            
            # Sắp xếp từ khóa theo sự thay đổi phần trăm (bao gồm cả giá trị âm)
            sorted_changes = sorted(changes.items(), key=lambda x: x[1], reverse=True)

            file.write(f"Ngày {date} so với ngày gốc {base_date}:\n")
            for keyword, change in sorted_changes:
                file.write(f"- {keyword}: thay đổi {change:.2f}%\n")
            file.write("\n")
