import py_vncorenlp
from underthesea import chunk , word_tokenize
from pyvi import ViTokenizer, ViPosTagger
from transformers import AutoTokenizer , AutoModel
import torch
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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
with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
    stop_words = f.read().splitlines()

# Văn bản cần xử lý
text = "Hamas tuyên bố, sáng sớm nay (31/10) chiến binh của họ đã bắn tên lửa chống tăng về phía lực lượng Israel ở phía bắc và phía nam Gaza khi xe tăng và bộ binh của Israel tấn công thành phố chính của khu vực này."

# Loại bỏ từ dừng
filtered_text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

# py_vncorenlp.download_model(save_dir='C:\\Users\\Admin\\Downloads\\vncorenlp\\VnCoreNLP')
# model = py_vncorenlp.VnCoreNLP(save_dir='C:\\Users\\Admin\\Downloads\\vncorenlp\\VnCoreNLP')
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='C:\\Users\\Admin\\Downloads\\vncorenlp\\VnCoreNLP')
text_process = standardize_data(filtered_text)
phobert = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

seg_text = rdrsegmenter.word_segment(text_process)[0]
# seg_text = ' '.join([word for word in seg_text.split() if word.lower() not in stop_words])

words = seg_text.split()  # Tách từ dựa trên dấu cách

# Mã hóa cả câu
# input_ids = phobert_tokenizer.encode(seg_text, return_tensors="pt")
input_ids = torch.tensor([tokenizer.encode(seg_text )])
print(input_ids)
with torch.no_grad():
    outputs = phobert(input_ids)
    
# Trích xuất last hidden state
last_hidden_states = outputs.last_hidden_state.squeeze(0)

# Ánh xạ từ input_ids sang token
token_list = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

# Tìm vector tương ứng với từng từ trong 'words'
word_to_vec_map = {}
for word in words:
    # Initialize an empty list collect the states for sub-tokens
    subtoken_states = []
    
    # Loop through each token to check if it's a part of the current word
    for i, token in enumerate(token_list):
        if word in token:
            subtoken_states.append(last_hidden_states[i])
            
    # If any sub-tokens were found, average their states
    if len(subtoken_states) > 0:
        word_vec = torch.mean(torch.stack(subtoken_states), dim=0)

        word_to_vec_map[word] = word_vec
    else:
        print(f"Word {word} was not found in the token list.")

# Tiếp tục xử lý với 'word_to_vec_map'

# for word, vec in zip(words, word_vectors):
#     print(f"Token: {word}")
#     print(f"Vector: {vec}")
# word_vectors_np = [vec.detach().numpy() for vec in word_vectors]

# document_embedding = np.mean(word_vectors_np, axis=0)
# keywords = []
# for word, word_vector in zip(words, word_vectors_np):
#     similarity = cosine_similarity(
#         [word_vector],
#         [document_embedding]
#     )
#     keywords.append((word, similarity))
# sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
# print(sorted_keywords)
# vectorizer = TfidfVectorizer(analyzer=lambda x: x.split())
# tfidf_matrix = vectorizer.fit_transform([' '.join(words)])
# tfidf_feature_names = vectorizer.get_feature_names_out()
# tfidf_scores = tfidf_matrix.toarray().flatten()

# # Create a dictionary to map words to their TF-IDF scores
# tfidf_dict = dict(zip(tfidf_feature_names, tfidf_scores))

# # Get the TF-IDF scores corresponding to each word in the list `words`
# tfidf_scores_for_words = [tfidf_dict.get(word, 0) for word in words]

# # Combine TF-IDF and PhoBERT
# enhanced_word_vectors = []
# for word, tfidf in zip(words, tfidf_scores_for_words):
#     if word in word_to_vec_map:
#         enhanced_word_vectors.append(word_to_vec_map[word] * tfidf)
#     else:
#         print(f"Word {word} not found in word_to_vec_map.")

# # Compute the new document embedding
# # Chuyển từ PyTorch tensor sang numpy array
# enhanced_word_vectors_np = [tensor.cpu().numpy() for tensor in enhanced_word_vectors]

# # Tính toán document embedding mới
# enhanced_document_embedding = np.mean(enhanced_word_vectors_np, axis=0)

# # Rank words based on cosine similarity to the document embedding
# keywords = []
# for word, word_vector in zip(words, enhanced_word_vectors_np):
#     similarity = cosine_similarity(
#         [word_vector],
#         [enhanced_document_embedding]
#     )
#     keywords.append((word, similarity))

# sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
# print(sorted_keywords)
# Chuyển các tensor trong word_to_vec_map thành mảng NumPy
word_to_vec_map_np = {word: vec.detach().cpu().numpy() for word, vec in word_to_vec_map.items()}

# Tính trung bình của tất cả các vectơ từ để có được document embedding
document_embedding = np.mean(list(word_to_vec_map_np.values()), axis=0)

# Xếp hạng các từ dựa trên độ tương tự cosine với document embedding
keywords = []
for word, word_vector in word_to_vec_map_np.items():
    similarity = cosine_similarity(
        [word_vector],
        [document_embedding]
    )
    keywords.append((word, similarity))

# Sắp xếp các từ dựa trên độ tương tự
sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)

# In ra các từ quan trọng
print(sorted_keywords)
