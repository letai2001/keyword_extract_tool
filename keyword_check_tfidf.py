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

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([seg_text])
tfidf_feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray().flatten()

# Map words to their TF-IDF scores
tfidf_dict = dict(zip(tfidf_feature_names, tfidf_scores))

# Select top 10 words based on TF-IDF scores
top_10_words = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:10]
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
for s in top_3_sentences:
    words_in_sentence = s.split()
    pos_tags_in_sentence = [pos_tags[tokens.index(w)] for w in words_in_sentence if w in tokens]

    # Combine words and their POS tags
    combined_data = list(zip(words_in_sentence, pos_tags_in_sentence))

    # Extract nouns with high TF-IDF values
    high_tfidf_nouns = [w for w, pos in combined_data if pos in ['N', 'Nc', 'Np', 'Nu'] and tfidf_dict.get(w, 0) in top_10_words]
    final_keywords.extend(high_tfidf_nouns)

    # Extract other nouns in the sentence
    other_nouns = [w for w, pos in combined_data if pos in ['N', 'Nc', 'Np', 'Nu'] and w not in high_tfidf_nouns]
    final_keywords.extend(other_nouns)

    # Extract adjectives and verbs with high TF-IDF values
    high_tfidf_adj_verbs = [w for w, pos in combined_data if pos in ['A', 'V'] and tfidf_dict.get(w, 0) in top_10_words]
    final_keywords.extend(high_tfidf_adj_verbs)

print(top_10_words, top_3_sentences, final_keywords)

