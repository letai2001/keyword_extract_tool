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
from vncorenlp import VnCoreNLP
number_load_pho = 6
with open('vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
    stop_words = f.read().splitlines()
def replace_(text):
    return re.sub(r'(?<=\w)-(?=\w)', '_', text)
text = "Ở Hà Nội , Đại biểu Huỳnh Thanh Phương (Bí thư Huyện ủy Gò Dầu, tỉnh Tây Ninh) nói tín dụng đen vẫn len lỏi không chỉ ở vùng nông thôn mà ngay cả thành thị, thậm chí một số nơi đang có dấu hiệu hoạt động mạnh trở lại. Ông đề nghị Thống đốc Ngân hàng Nhà nước cho biết nguyên nhân, nêu giải pháp ngăn chặn. Thống đốc Nguyễn Thị Hồng nói vấn đề này được Đảng, Nhà nước, Chính phủ, Quốc hội rất quan tâm. Bộ Công an, Ngân hàng Nhà nước có rất nhiều biện pháp để tăng khả năng tiếp cận tín dụng ở các kênh chính thức như hoàn thiện hành lang pháp lý tạo thuận lợi cho người dân tiếp cận tín dụng của các công ty tài chính và công ty tài chính tiêu dùng; sửa đổi quy định để người dân được cấp tín dụng qua các phương thức điện tử để tạo thuận lợi. Thời gian qua, Bộ Công an đã triển khai hệ thống dữ liệu dân cư quốc gia, Ngân hàng Nhà nước đã ký kế hoạch thúc đẩy tổ chức tín dụng tham gia hệ thống dữ liệu này để tiến tới cho vay tín chấp các khoản nhỏ lẻ để hạn chế tín dụng đen. Bà đề nghị UBND các cấp, tổ chức chính trị xã hội nắm bắt thông tin để giúp người dân có nhu cầu được vay vốn chính đáng, hạn chế phải tìm đến tín dụng đen."
pattern = re.compile(r'(?<!\b\w\.)\. (?=[A-Z])|\.$')


vn_core = VnCoreNLP("C:\\Users\\Admin\\Downloads\\vncorenlp\\VnCoreNLP\\VnCoreNLP-1.2.jar" ,  annotators="wseg,pos,ner , parse", max_heap_size='-Xmx2g')
vncorenlp_example = vn_core.tokenize(text)

print(vncorenlp_example)
print('POS Tagging:', vn_core.pos_tag(text))
combined_data = vn_core.pos_tag(text)
filtered_combined_data = [
    (token, pos_tag) for sublist in combined_data for token, pos_tag in sublist if token.lower() not in stop_words
]
# sentence_combined_data = [
#     [(token, pos_tag) for token, pos_tag in sublist if token.lower() not in stop_words]
#     for sublist in combined_data
# ]
print(filtered_combined_data)
# print(sentence_combined_data)
key_words = []
for w , pos in filtered_combined_data:
    #lay them ca dongtu ,  tinhtu
    # if((pos == 'Np' or (pos in ['N', 'Nc' , 'Ny' , 'Nb' , 'V' , 'A'  ] and '_' in w)) ):
    #     key_words.append(w)
    #chi lay them ca danhtu
    if((pos == 'Np' or (pos in ['N', 'Nc' , 'Ny' , 'Nb'  ] and '_' in w)) ):
        key_words.append(w)
key_words = [word.lower() for word in key_words ]

key_words = set(key_words)
print(key_words)        
    
    
# sentences = pattern.split(text)
# sentences = [s.strip() for s in sentences if s]  # Remove any leading/trailing whitespace
# seg_sentences = []
# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='C:\\Users\\Admin\\Downloads\\vncorenlp\\VnCoreNLP')

# # seg_sentences = [rdrsegmenter.word_segment(sentence)[0] for sentence in sentences]
# for sentence in sentences:
#     a = rdrsegmenter.word_segment(sentence)[0]
#     seg_sentences.append(a)
# seg_text = ". ".join(seg_sentences)
# model = phonlp.load(save_dir='C:\\absolute\\path\\to\\pretrained_phonlp')
# all_tokens = []
# all_pos_tags = []
# sentence_groups = [seg_sentences[i:i + number_load_pho] for i in range(0, len(seg_sentences), number_load_pho)]
# for group in sentence_groups:
#     # Combine group into a single string to annotate
#     group_text = ' '.join(group)
#     annotated_data = model.annotate(text=group_text)
    
#     # Extract tokens and POS tags from the annotated data
#     for sentence in annotated_data[0]:
#         tokens = sentence
#         all_tokens.extend(tokens)
    
#     for sentence in annotated_data[1]:
#         pos_tags = [tag[0] for tag in sentence]
#         all_pos_tags.extend(pos_tags)

# # Combine the tokens and POS tags into a single list of tuples
# combined_data = list(zip(all_tokens, all_pos_tags))
# seg_text = ' '.join([word for word in seg_text.split() if word.lower() not in stop_words])
# filtered_combined_data = [(token, pos_tag) for token, pos_tag in combined_data if token.lower() not in stop_words]

# key_words = []
# for w , pos in filtered_combined_data:
#     if((pos == 'Np' or (pos in ['N', 'Nc' , 'Ny' , 'Nb'  ] and '_' in w)) ):
#         key_words.append(w)
# key_words = [word.lower() for word in key_words ]

# key_words = set(key_words)
# print(key_words)        


# combined_data_ner = vn_core.ner(text)
# dep_parses = vn_core.dep_parse(text)
# print(f)
# important_labels = ['root', 'nmod', 'vmod', 'sub', 'iob', 'pob']

# # Hàm để trích xuất từ khóa từ mỗi câu
# def extract_keywords(dep_parse, tokens):
#     keywords = []
#     for dep in dep_parse:
#         label, head_idx, dep_idx = dep
#         if label in important_labels:
#             keyword = tokens[dep_idx - 1]  # Chỉ số trong Python bắt đầu từ 0
#             if keyword not in keywords:
#                 keywords.append(keyword)
#     return keywords

# # Duyệt qua mỗi câu và trích xuất từ khóa
# all_keywords = []
# for i, dep_parse in enumerate(dep_parses):
#     keywords = extract_keywords(dep_parse, vncorenlp_example[i])
#     all_keywords.extend(keywords)
# print(all_keywords)
    
