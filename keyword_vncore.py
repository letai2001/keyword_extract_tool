import py_vncorenlp
from underthesea import chunk , word_tokenize
from pyvi import ViTokenizer, ViPosTagger
from transformers import AutoTokenizer , AutoModel
import torch
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import pos_tag
import phonlp
def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row
with open('vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
    stop_words = f.read().splitlines()

# Văn bản cần xử lý
text = "Hỗ trợ phân tích các chuẩn Log phổ biến hiện nay, tập trung vào vấn đề giám sát an ninh, hỗ trợ cảnh báo qua Email và SMS”"

# Loại bỏ từ dừng
filtered_text = standardize_data(text)
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='C:\\Users\\Admin\\Downloads\\vncorenlp\\VnCoreNLP')

# Phân tách văn bản thành từng câu
sentences = text.split(". ")

# Thực hiện word segmentation cho từng câu
seg_sentences = [rdrsegmenter.word_segment(sentence)[0] for sentence in sentences]

# Gộp các câu đã được word segmentation lại
seg_text = ". ".join(seg_sentences)

# seg_text = ' '.join([word for word in seg_text.split() if word.lower() not in stop_words])
print(seg_text)
model = phonlp.load(save_dir='C:\\absolute\\path\\to\\pretrained_phonlp')

# Annotate a corpus where each line represents a word-segmented sentence
# model.annotate(input_file='/absolute/path/to/input.txt', output_file='/absolute/path/to/output.txt')

# Annotate a word-segmented sentence
data = model.annotate(text=seg_text)
tokens = data[0][0]
pos_tags = [tag[0] for tag in data[1][0]]

# Kết hợp hai danh sách:
combined_data = list(zip(tokens, pos_tags))
print(combined_data)
# # In kết quả
list_accept = ['N', 'Np', 'Nc', 'Nu', 'A', 'R', 'V', 'M']
break_list = ['C', 'CH', 'F']

key_word = []
data_key_word = ""

for i, token in enumerate(combined_data):
    key_index = token[0]
    key_type_index = token[1]

    if key_type_index in list_accept and len(key_index) > 1 and key_index.find("http") < 0:
        data_key_word += (" " if data_key_word else "") + key_index

        # If the next token type is in break_list, we append and reset the data_key_word.
        if i+1 < len(combined_data) and combined_data[i+1][1] in break_list:
            key_word.append(data_key_word.lower())
            data_key_word = ""
    else:
        if data_key_word:
            key_word.append(data_key_word.lower())
        data_key_word = ""

if data_key_word:
    key_word.append(data_key_word.lower())

key_word_value = ", ".join([value.lower().replace(" ", "_") for value in key_word])
print(key_word_value)

# list_accept = ['N', 'Nc', 'Ny', 'Np', 'Nu', 'A']
# break_list = ['Ny', 'Np']
# key_word = []
# data_key_word = ""
# type_data = ""

# for i in range(len(pos_tags_list)):
#     word, pos_tag = pos_tags_list[i]
    
#     if pos_tag in list_accept:
#         if not data_key_word:
#             data_key_word += word
#             type_data += pos_tag
#         else:
#             data_key_word += " " + word
#             type_data += " " + pos_tag
        
#         # Check the next token's pos_tag
#         if data_key_word and pos_tag in break_list and i < (len(pos_tags_list) - 1) and pos_tags_list[i + 1][1] not in break_list:
#             if data_key_word not in key_word:
#                 key_word.append(data_key_word)
#             data_key_word = ""
#             type_data = ""
#     else:
#         if data_key_word and ("Np" in type_data or "Ny" in type_data):
#             if data_key_word not in key_word:
#                 key_word.append(data_key_word)
#         data_key_word = ""
#         type_data = ""

# # Convert key words to the desired format
# key_word_value = []
# for value in key_word:
#     formatted_value = value.lower().replace("-", "_").replace("(", "").replace(")", "").replace(" ", "_")
#     if formatted_value not in key_word_value:
#         key_word_value.append(formatted_value)

# final_key_words = ", ".join(key_word_value)
# print(final_key_words)


