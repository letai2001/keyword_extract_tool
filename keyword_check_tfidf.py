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
text = "Chủ tịch nước Võ Văn Thưởng cho biết, Việt Nam và Mông Cổ hoan nghênh việc ký kết hiệp định về miễn thị thực cho người dân hai nước, góp phần mở rộng hợp tác thương mại, du lịch, giao lưu nhân dân. Đồng thời, hai bên hoan nghênh những kết quả đạt được trong hợp tác quốc phòng - an ninh trong thời gian qua, nhất trí tăng cường hợp tác thực chất trong lĩnh vực tư pháp, công nghiệp quốc phòng, an ninh, thông qua những dự án cụ thể, trong đó có hợp tác chống tội phạm ma túy xuyên quốc gia, trao đổi học viên quốc phòng và an ninh. Trong chuyến thăm lần này của Tổng thống Mông Cổ, Bộ trưởng Ngoại giao hai nước đã ký Hiệp định giữa Việt Nam và Mông Cổ về miễn thị thực cho hộ chiếu ngoại giao, hộ chiếu công vụ, hộ chiếu phổ thông. ệp định về miễn thị thực cho người dân hai nước, góp phần mở rộng hợp tác thương mại, du lịch, giao lưu nhân dân. Đồng thời, hai bên hoan nghênh những kết quả đạt được trong hợp tác quốc phòng - an ninh trong thời gian qua, nhất trí tăng cường hợp tác thực chất trong lĩnh vực tư pháp, công nghiệp quốc phòng, an ninh, thông qua những dự án cụ thể, trong đó có hợp tác chống tội phạm ma túy xuyên quốc gia, trao đổi học viên quốc phòng và an ninh. Trong chuyến thăm lần này của Tổng thống Mông Cổ, Bộ trưởng Ngoại giao hai nước đã ký Hiệp định giữa Việt Nam và Mông Cổ về miễn thị thực cho hộ chiếu ngoại giao, hộ chiếu công vụ, hộ chiếu phổ thông. Chủ tịch nước Võ Văn Thưởng cũng cho biết, hai nước nhất trí thúc đẩy mục tiêu nâng gấp đôi kim ngạch thương mại song phương lên mức 200 triệu USD trong những năm tới, tiếp tục thúc đẩy mở rộng xuất nhập khẩu hàng hóa song phương, mở cửa cho hàng hóa của nhau trên cơ sở đáp ứng các tiêu chuẩn, nhu cầu của mỗi bên. Tăng cường các hoạt động xúc tiến đầu tư, đẩy mạnh hợp tác, đa dạng hóa chuỗi cung ứng nguyên liệu, nhất là trong các ngành hai bên có thế mạnh. Nghiên cứu khả năng hợp tác trong lĩnh vực khoáng sản chiến lược giữa hai nước và hỗ trợ doanh nghiệp hai nước trong quá trình nghiên cứu, triển khai hoạt động đầu tư, đẩy mạnh hợp tác giữa các địa phương hai nước, thiết lập các cặp quan hệ hợp tác về kinh tế. Chủ tịch nước Võ Văn Thưởng cũng cho biết, hai nước nhất trí tiếp tục tạo điều kiện và bảo vệ quyền lợi chính đáng cho công dân hai nước yên tâm sinh sống, học tập và làm việc tại nước kia, phát huy vai trò cầu nối, thúc đẩy quan hệ hữu nghị Việt Nam - Mông Cổ. Trên bình diện đa phương, Chủ tịch nước Võ Văn Thưởng nhấn mạnh, Việt Nam mong muốn Mông Cổ tiếp tục ủng hộ lập trường của ASEAN trong bảo đảm an ninh, an toàn hàng hải, hàng không ở Biển Đông, ủng hộ giải quyết mọi tranh chấp, bất đồng bằng biện pháp hòa bình, trên cơ sở luật pháp quốc tế, nhất là Công ước của Liên Hiệp Quốc về luật Biển năm 1982."

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
seg_text = ' '.join([word for word in seg_text.split() if word.lower() not in stop_words])


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
# high_tfidf_nouns = [
#     w for w, pos in combined_data
#     if (pos == 'Np' or (pos in ['N', 'Nc', 'Nu'] and '_' in w)) and w.lower() in top_10_words
# ]
# final_keywords.extend(high_tfidf_nouns)
sorted_tfidf_example = sorted(filtered_tfidf_dict.items(), key=lambda x: x[1], reverse=True)
words_from_sorted = [word for word, score in sorted_tfidf_example]

# Extract separate lists for nouns, verbs, and adjectives
nouns_example_10 = [word for word, tag in combined_data if tag in ['N', 'Nc', 'Np', 'Nu'] and word in words_from_sorted][:10]
verbs_adjectives_example_5 = [word for word, tag in combined_data if tag in ['V', 'A'] and word in words_from_sorted][:5]
nouns_example_set = set(nouns_example_10)
verbs_adjectives_example_set = set(verbs_adjectives_example_5)

# Get the top 10 nouns and top 5 verbs/adjectives
# top_10_nouns_example = sorted(nouns_example_set, key=lambda w: sorted_tfidf_example[w], reverse=True)[:10]
# top_5_verbs_adjectives_example = sorted(verbs_adjectives_example_set, key=lambda w: sorted_tfidf_example[w], reverse=True)[:5]
final_keywords.extend(nouns_example_set)
final_keywords.extend(verbs_adjectives_example_set)


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
    
#     high_tfidf_adj_verbs = [
#     w for w, pos in combined_data 
#     if ('_' in w and pos in ['A', 'V'] and tfidf_dict.get(w.lower(), 0) in top_10_words)
# ]
#     final_keywords.extend(high_tfidf_adj_verbs)

final_keywords = set(final_keywords)
print(top_10_words, top_3_sentences, final_keywords)


