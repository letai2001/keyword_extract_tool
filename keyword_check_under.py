from underthesea import chunk , word_tokenize
from transformers import PhobertTokenizer

data_check = "Ông Phan Văn Quý, người lính trực tiếp chiến đấu tại tuyến đường Trường Sơn lịch sử, đại diện nhóm cựu chiến binh cho biết: “Lời nói của các Mẹ đã thôi thúc chúng tôi, những cựu chiến binh may mắn được trở về sau chiến tranh phải làm điều gì đó để đưa đồng đội mình về với gia đình, người thân. Vì thế, một nhóm cựu chiến binh đã đưa ra ý tưởng thành lập “Quỹ hỗ trợ giám định ADN hài cốt liệt sỹ” để cùng chung tay với Nhà nước ta trong việc thực hiện công tác trả lại tên cho các liệt sỹ”."
phobert_tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")
tokens = phobert_tokenizer.encode(data_check)
tagged_tokens = chunk(" ".join(tokens))

# Sử dụng underthesea để tách từ và gán các loại từ ngữ
tokens = chunk(data_check)

list_accept = ['N', 'Np', 'Nc', 'Nu', 'A', 'R', 'V', 'M']
break_list = ['C', 'CH', 'F']

key_word = []
data_key_word = ""

for token in tagged_tokens:
    key_index = token[0]
    key_type_index = token[1]

    if key_type_index in list_accept and len(key_index) > 1 and key_index.find("http") < 0:
        data_key_word += (" " if data_key_word else "") + key_index

        # If the next token type is in break_list, we append and reset the data_key_word.
        if token[2] in break_list:
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
