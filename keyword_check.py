from pyvi import ViTokenizer, ViPosTagger
import re

data_check = "Theo tin từ Đài khí tượng thủy văn Trung ương, hồi 13 giờ ngày 10/12, vị trí tâm bão số 9 ở khoảng 14,1 độ Vĩ Bắc; 111,7 độ Kinh Đông, cách đảo Song Tử Tây khoảng 210km về phía Bắc Đông Bắc."

key_words_info = ViPosTagger.postagging(ViTokenizer.tokenize(data_check.replace("-","_").replace("(","").replace(")","")))
print(key_words_info)
list_accept =  ['N', 'Nv', 'Nc', 'Nu', 'A', 'R', 'V' ,'M']
break_list = [ 'C', 'CH' , 'F']

key_word = []
data_key_word = ""

for idx in range(len(key_words_info[0])):
    key_index = key_words_info[0][idx]
    key_type_index = key_words_info[1][idx]
    
    if key_type_index in list_accept and len(key_index) > 1 and key_index.find("http") < 0:
        data_key_word += (" " if data_key_word else "") + key_index
        
        # If the next word type is in break_list, we append and reset the data_key_word.
        if idx < len(key_words_info[1]) - 1 and key_words_info[1][idx+1] in break_list:
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
