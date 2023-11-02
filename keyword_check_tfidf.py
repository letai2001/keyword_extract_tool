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
number_load_pho = 6
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
def replace_(text):
    return re.sub(r'(?<=\w)-(?=\w)', '_', text)

# def annotate_and_combine(sentences, model):
#     # Initialize lists to hold the combined results

# return combined_annotated_data

# Văn bản cần xử lý
text = "Hãng Reuters đưa tin, Israel đã mở rộng các hoạt động trên bộ ở Gaza nhằm trả đũa Hamas - lực lượng nắm quyền tại Dải Gaza, vì cuộc tấn công bất ngờ vào nước này cách đây 3 tuần, làm 1.400 người thiệt mạng. Các nhân chứng cho biết, lực lượng Israel nhắm mục tiêu vào tuyến đường chính nối bắc và nam Gaza và tấn công Gaza City từ hai hướng. Lữ đoàn Al-Qassam, cánh vũ trang của Hamas cho hay, sáng sớm nay các chiến binh của họ đụng độ với quân Israel đang tràn vào trục phía nam Gaza. Chiến binh Hamas đã dùng súng máy và các tên lửa al-Yassin 105 để nhắm bắn các phương tiện của Israel. Ngoài ra, họ cũng dùng tên lửa để hạ hai xe tăng và nhiều xe ủi đất khác của Israel tại tây bắc Gaza, lữ đoàn Al-Qassam cho hay. Israel hiện chưa bình luận gì về thông tin Hamas đưa ra và Reuters cho biết hiện chưa thể xác nhận thông tin về giao tranh. Theo giới chức y tế Gaza, tổng số 8.306 người đã thiệt mạng kể từ khi Israel tấn công khu vực này. Trong khi đó, các quan chức Liên Hợp Quốc nói hơn 1,4 triệu người trong tổng số 2,3 triệu dân ở Gaza đã mất nhà.Số người thiệt mạng do xung đột Israel - Hamas tăng đã khiến Mỹ và nhiều quốc gia khác kêu gọi tạm dừng giao tranh để có thể đưa viện trợ nhân đạo tới Gaza. Các chuyên gia quân sự nhận xét, lực lượng Israel đang tấn công trên bộ một cách chậm rãi, một phần là để ngỏ khả năng các chiến binh Hamas sẽ đàm phán thả con tin. Điều này trái ngược với những cuộc không kích không ngừng nghỉ trong 3 tuần qua cũng như các cuộc tấn công trên bộ trước đây của Israel vào Dải Gaza."

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
all_tokens = []
all_pos_tags = []
sentence_groups = [seg_sentences[i:i + number_load_pho] for i in range(0, len(seg_sentences), number_load_pho)]

# Iterate over each sentence and annotate it
for group in sentence_groups:
    # Combine group into a single string to annotate
    group_text = ' '.join(group)
    annotated_data = model.annotate(text=group_text)
    
    # Extract tokens and POS tags from the annotated data
    for sentence in annotated_data[0]:
        tokens = sentence
        all_tokens.extend(tokens)
    
    for sentence in annotated_data[1]:
        pos_tags = [tag[0] for tag in sentence]
        all_pos_tags.extend(pos_tags)

# Combine the tokens and POS tags into a single list of tuples
combined_data = list(zip(all_tokens, all_pos_tags))


# Annotate a corpus where each line represents a word-segmented sentence
# model.annotate(input_file='/absolute/path/to/input.txt', output_file='/absolute/path/to/output.txt')
# combined_data = annotate_and_combine(sentences, model)

# Annotate a word-segmented sentence
# data = model.annotate(text=seg_text)

# tokens = data[0][0]
# pos_tags = [tag[0] for tag in data[1][0]]

# # Kết hợp hai danh sách:
# combined_data = list(zip(tokens, pos_tags))
print(combined_data)
def custom_tokenizer(text):
    # Use a regular expression to split the text into tokens
    return re.findall(r'\b\w+\b', text)  # This regex will split the words at spaces

vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
seg_text = replace_(seg_text)
sentences = seg_text.split(". ")


tfidf_matrix = vectorizer.fit_transform([seg_text])

tfidf_feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray().flatten()

# Map words to their TF-IDF scores
tfidf_dict = dict(zip(tfidf_feature_names, tfidf_scores))

# Select top 10 words based on TF-IDF scores
# top_10_words = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:10]
# top_10_words = [word for word, score in top_10_words]
filtered_tfidf_dict = {
    word: score for word, score in tfidf_dict.items()
    if '_' in word or any(tag == 'Np' and w.lower() == word for w, tag in combined_data)
}
# Select top 10 words based on the filtered TF-IDF scores
top_10_words = sorted(filtered_tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:10]
top_10_words = [word for word, score in top_10_words]

# Step 2: Identify important sentences
important_sentences = [s for s in seg_sentences if any(word in s for word in top_10_words)]

# Step 3: Select the top 3 important sentences based on their total TF-IDF score
sentence_tfidf_scores = []
for s in important_sentences:
    words_in_sentence = s.split()
    sentence_tfidf_score = sum(tfidf_dict.get(w, 0) for w in words_in_sentence)
    sentence_tfidf_scores.append((s, sentence_tfidf_score))

# Sort the sentences by their TF-IDF scores
sorted_sentences = sorted(sentence_tfidf_scores, key=lambda x: x[1], reverse=True)[:3]

# Extract the top 3 important sentences
top_3_sentences = [s[0] for s in sorted_sentences]

# Step 4: Extract keywords from the top 3 important sentences
final_keywords = []
high_tfidf_nouns = []
high_tfidf_nouns = [
    w for w, pos in combined_data
    if (pos == 'Np' or (pos in ['N', 'Nc', 'Nu'] and '_' in w)) and w.lower() in top_10_words
]
# for w  , pos in combined_data: 
#     k = w.lower()
#     h = tfidf_dict.get(k, 0)
#     if (pos == 'Np' or (pos in ['N', 'Nc', 'Nu'] and '_' in w)) and h in top_10_words:
#        high_tfidf_nouns.append(w) 

final_keywords.extend(high_tfidf_nouns)
for s in top_3_sentences:
    words_in_sentence = s.split()
    pos_tags_in_sentence = [all_pos_tags[all_tokens.index(w.lower())] for w in words_in_sentence if w.lower() in all_tokens]

    # Combine words and their POS tags
    combined_data = list(zip(words_in_sentence, pos_tags_in_sentence))

    # Extract nouns with high TF-IDF values that are compound words with specific POS tags or proper nouns
    # high_tfidf_nouns = [
    #     w for w, pos in combined_data 
    #     if ((pos == 'Np' or (pos in ['N', 'Nc', 'Nu'] and '_' in w)) and tfidf_dict.get(w.lower(), 0) in top_10_words)
    # ]
    # final_keywords.extend(high_tfidf_nouns)

    # Extract other compound words with specific POS tags and proper nouns in the sentence
    other_nouns = [
        w for w, pos in combined_data 
        if ((pos == 'Np' or (pos in ['N', 'Nc', 'Nu'] and '_' in w)) and w.lower() not in [word.lower() for word in high_tfidf_nouns])
    ]
    final_keywords.extend(other_nouns)
    
    high_tfidf_adj_verbs = [
    w for w, pos in combined_data 
    if ('_' in w and pos in ['A', 'V'] and tfidf_dict.get(w.lower(), 0) in top_10_words)
]
    final_keywords.extend(high_tfidf_adj_verbs)


print(top_10_words, top_3_sentences, final_keywords)


