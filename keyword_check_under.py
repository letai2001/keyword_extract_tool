import numpy as np
import re
from vncorenlp import VnCoreNLP
import json
with open('vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
    stop_words = f.read().splitlines()
    
with open('content_test_media.json', 'r' , encoding='utf-8') as file:
    data = json.load(file)

vn_core = VnCoreNLP("C:\\Users\\Admin\\Downloads\\vncorenlp\\VnCoreNLP\\VnCoreNLP-1.2.jar" ,  annotators="wseg,pos ,ner", max_heap_size='-Xmx2g')

# text = """tôi hy vọng mn thấy ra được giá trị Trung Quốc theo đuổi là chủ nghĩa đại Hán. Họ căm ghét Nhật Bản nhưng tư tưởng của họ cũng chả khác gì Nhật Bản. Nhật thời phát xít cũng coi người Nhật Bản là ưu việt và muốn phát quang ra bên ngoài với biểu tượng là lá cờ có các gạch đỏ như mặt trời chiếu rọi nhân gian. Tôi vd như Pháp, Mỹ cũng từng xâm lược Việt Nam và bất kể chính phủ họ có lòe bịp dân họ như thế nào thì dân họ luôn có những người tin vào những giá trị Pháp tự do bình đẳng bác ái hay giá trị Mỹ tự do dân chủ nhân quyền và họ sẽ chiến đấu vào các giá trị đó không coi đó là bình phòng, họ mới chính là những người trung thành nhất với nước Pháp, nước Mỹ. Tương tự, VN cũng có giá trị mình theo đuổi là độc lập tự do hạnh phúc, các vị có thể thấy cách VN hành xử trong cuộc chiến chống xâm lược diệt chủng pol pot. VN đã ko thèm làm trò quỷ như TQ và Mỹ là chia đôi Campuchia ra để có vùng đệm ct. Trung Quốc chỉ theo đuổi một giá trị là chủ nghĩa đại Hán, nếu không phải các vị nói tôi nghe họ theo đuổi giá trị gì. 
# Hiện Tập Cận Bình vẫn là tổng bí thư nhưng bất kể có phải ông ta hay không thì giá trị TQ theo đuổi vẫn mãi mãi là chủ nghĩa đại Hán. VD các vị để ý trong tất cả các phim cổ trang của TQ, họ luôn dùng một thái độ thống nhất là các nước dân tộc ở phía Bắc TQ là những kẻ phản diện và TQ đã đánh là luôn thắng và chả thấy chính phủ TQ hay dân TQ la trời la đất là bóp méo lịch sử bao giờ. SỰ ĐỒNG THUẬN TRƠ TRÁO của dân TQ VÀ SỰ PHÂN BIỆT CHỦNG TỘC NGƯỜI HÁN LÀ TUYỆT VỜI NHẤT CỦA HỌ ĐẤY, CÁC VỊ THẤY CHƯA. Nhưng sự thật lịch sử là chính các nước các dân tộc như Kim, Mông, Mãn đã thôn tính TQ đấy chứ, vậy xin hỏi ai thắng đây. Sự lãng quên và sự phủ nhận của chính phủ như thế rất vô liêm sỉ và gây mất niềm tin không chỉ trong đối ngoại mà còn đối nội nữa.
# Đánh giá về người khác không nên theo theo kiểu bụng ta suy ra bụng người, câu này thường dùng theo nghĩa tiêu cực nhưng ở đây ý tôi rằng không nên vì mình xấu dở tồi v.v... mà nghĩ người ta xấu dở tồi v.v... và không nên vì mình tốt giỏi v.v... mà cũng nghĩ họ tốt giỏi. Điều này chỉ khiến các vị đánh giá người và sự với quá nhiều cảm xúc thiên vị, cho nên dựa trên chính biểu hiện của họ để đánh giá mới chính xác. người Việt Nam luôn hòa hiếu và lấy cái nền tảng ấy mà đi suy xét bụng dạ Trung Quốc để rồi bị lừa, bị lợi dụng. 
# Khi Tần Thủy Hoàng xâm lược và thôn tính khắp nơi gồm cả tổ tiên ta, có ai dám nói triều đại của ông ta sẽ sụp đổ cùng ông ta đâu. Tần Thủy Hoàng ngoài xâm lược còn có bình định và làm rất tốt nữa, lãnh thổ không rộng lớn như của Thành Cát Tư Hãn mà còn sụp đổ chóng vánh hơn đế quốc Mông Cổ. Ối dời ôi. Hán Vũ Đế một vị hoàng đế hoành tráng khác của nhà Hán sau này cũng xâm lược VN và được TQ tung hô như minh quân và sau đó thế nào, lịch sử TQ còn tên hoàng đế nhà Hán nào đáng kể nữa không. LỊCH SỬ TRUNG QUỐC SAU ĐÓ ĐƯỢC VIẾT NÊN BỞI CÁC DÂN TỘC KHÔNG PHẢI HÁN TỘC NHƯNG ĐỊA VỊ CỦA HỌ TRONG DÒNG CHẢY LỊCH SỬ TRUNG QUỐC THÌ NHƯ THẾ NÀO. Triều đại phong kiến cuối cùng là triều Mãn Thanh không phải triều người Hán. Người Hán giỏi và tự tôn đến thế thì sao họ không biết kháng chiến để giành lại đất nước của mình đi khi triều Mãn Thanh suy vi. Họ không làm nổi. Triều Mãn Thanh đã bị phương Tây xóa sổ và người Hán trỗi lên. Vậy theo một cách cười ra nước mắt thì có phải họ nên cảm ơn phương Tây đã xâm lược TQ không. Họ không hề. Được rồi, cũng dễ hiểu. Vậy triều Mãn Thanh họ tự nhận đây là một triều đại của Trung Quốc và tự hào. Ủa, sao kỳ ta, người Mãn Thanh cũng xâm lược mà và người Hán nếu thật coi trọng người Mãn Thanh thì sao vd sao lại không tôn vinh sườn xám như quốc phục, họ không chịu thế. Các vị đã hiểu chưa. 
# Đối chiếu với Việt Nam, các vị có biết Hồ Quý Ly có tổ tiên là Hồ Hưng Dật vốn người Chiết Giang (Trung Quốc), đời Hậu Hán thời Ngũ đại Thập quốc (năm 947-950), tương đương thời Dương Tam Kha của Việt Nam, sang làm Thái thú Diễn Châu và định cư ở hương Bào Đột, nay là xã Quỳnh Lâm, huyện Quỳnh Lưu, tỉnh Nghệ An. Đến đời nhà Lý, có người trong họ lấy công chúa Nguyệt Đích, sinh ra công chúa Nguyệt Đoan. Đời cháu thứ 12 là Hồ Liêm dời đến ở hương Đại Lại, Thanh Hóa, làm con nuôi tuyên úy Lê Huấn, từ đấy lấy Lê làm họ mình. Quý Ly là cháu bốn đời của cụ Hồ Liêm. VÀ LỊCH SỬ VIỆT NAM LUÔN CÔNG NHẬN ÔNG LÀ MỘT NGƯỜI VIỆT NAM, Việt Nam ta không có cái thứ phân biệt chủng tộc dù theo màu da hay đại Kinh, đại Tày, đại Hán, đại Khmer v.v... Đơn giản chúng ta là người Việt Nam mà thôi. nói về phân biệt chủng tộc, so sánh giữa Mỹ và TQ thì mn sẽ thấy được sự khác biệt giữa 2 trường hợp là Mỹ phân biệt chủng tộc theo màu da, TQ phân biệt chủng tộc theo kiểu như thuyết ưu sinh của Hitler vậy, ai là người hán thì tuyệt vời còn lại là rác rưởi. Người Việt có huyênh hoang chủ nghĩa đại Kinh, đại Tày, đại Nùng, v.v... như thế bao giờ đâu. 
# Tại VN không có sự phân biệt chủng tộc, chúng ta tự hào là người Việt Nam và trong đó gồm nhiều dân tộc kể cả người gốc hoa, gốc khmer v.v... Trung Quốc nói Trung Quốc là của người Hán, sau này bắt chước Việt Nam mới mị dân rằng TQ có 56 dân tộc anh em chứ trước đó chính phủ và dân Trung Quốc (tức người Hán) coi người ta là thứ hạ cấp, không như Việt Nam xem nhau là đồng bào bầu ơi thương lấy bí cùng đâu. Chính sự phân biệt chủng tộc như vậy tại TQ mà lắm kẻ muốn nhận mình là người Hán để yên ổn khỏi những tên người Hán khác hoặc để cảm thấy oách, kiểu thấy sang bắt quàng làm họ ấy chứ đặc quyền như đất đai hay cư trú thì không có. Thời Mãn Thanh, người Mãn có đặc quyền nên không phải ông muốn nhận là người Mãn là được đâu, ông có cha là người Mãn mà mẹ là người Hán thì ông vẫn không được nhìn nhận là người Mãn. Theo thời gian và sự biến thiên của lịch sử, số lượng tự nhận là người Hán càng nhiều và người Hán cũng không còn được xem là công dân hạng 2, bị đuổi giết vd như thời Càn Long, mà còn nảy nòi ra chủ thuyết phục hồi chủ nghĩa đại Hán. Nếu không từng biến mất, sao lại phải phục hồi!!! Những lời nói dối luôn tự mâu thuẫn với nhau, mn nghe Trung Quốc tuyên truyền mà xem"""
# def combine_paragraphs(text):
#     """
#     Combine multiple paragraphs in a text into a single, cohesive paragraph.

#     :param text: String containing multiple paragraphs.
#     :return: A single combined paragraph.
#     """
#     # Splitting the text into paragraphs
#     paragraphs = text.split("\n")

#     # Removing any extra spaces and empty lines
#     cleaned_paragraphs = [para.strip() for para in paragraphs if para.strip()]

#     # Combining the paragraphs into a single string
#     combined_paragraph = " ".join(cleaned_paragraphs)

#     return combined_paragraph
key_words_dict = {}
def add_to_json_file(file_path, key_words_dict):
    try:
        # Đọc file hiện tại và chuyển nó thành một dictionary
        with open(file_path, 'r', encoding='utf-8') as file:
            current_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # Nếu file không tồn tại hoặc trống
        current_data = {}

    # Cập nhật dữ liệu mới vào dictionary hiện tại
    current_data.update(key_words_dict)

    # Ghi lại dữ liệu vào file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(current_data, file, ensure_ascii=False, indent=4)
def clean_and_format_text(text):
    # Xóa các ký tự đặc biệt (không xóa dấu chấm và dấu phẩy)
    cleaned_text = re.sub(r'[^\w\s,.]', '', text)

    # Đảm bảo có khoảng trắng sau mỗi dấu chấm
    formatted_text = re.sub(r'\.([^\s])', r'. \1', cleaned_text)

    return formatted_text

def extract_noun(key_words_info):
    list_accept = ['N', 'Nc', 'Ny', 'Np', 'Nu', 'A']
    break_list = ['Ny', 'Np']
    key_word = []
    data_key_word = ""
    type_data = ""

    for word, word_type in key_words_info:
        if word_type in list_accept:
            if data_key_word == "":
                data_key_word = word
                type_data = word_type
            else:
                data_key_word += " " + word               
                type_data += " " + word_type

            # Kiểm tra điều kiện để phân tách cụm từ khóa
            if data_key_word != "" and word_type in break_list and key_words_info.index((word, word_type)) < (len(key_words_info) - 1) and key_words_info[key_words_info.index((word, word_type)) + 1][1] not in break_list:
                if data_key_word not in key_word:
                    key_word.append(data_key_word)
                data_key_word = ""
                type_data = ""
        else:
            if data_key_word != "" and ("Np" in type_data or "Ny" in type_data):
                if data_key_word not in key_word:
                    key_word.append(data_key_word)
            data_key_word = ""
            type_data = ""
                                        
    # Xử lý và lưu trữ các từ khóa
    key_word_value = []
    for value in key_word:
        value = value.lower().replace("-", "_").replace("(", "").replace(")", "").replace(" ", "_")
        if value not in key_word_value:
            key_word_value.append(value)
    return key_word_value
for item in data:
    text = item['_source']['content']
    text = clean_and_format_text(text)
    print('đang trích keyword : ')
    print(f"ID: {item['_id']}")
    # print('POS Tagging:', vn_core.pos_tag(text))
    combined_data = vn_core.pos_tag(text)
    filtered_combined_data = [
        (token, pos_tag) for sublist in combined_data for token, pos_tag in sublist if token.lower() not in stop_words
    ]
    print(filtered_combined_data)
    # ner_data = vn_core.ner(text)
    # key_word_np = list(set([word for sublist in ner_data for word, tag in sublist if tag != 'O']))

    # print(sentence_combined_data)
    key_words = []
    key_words_np = []
    for w , pos in filtered_combined_data:
        # if((pos in ['Np' , 'Ny' , 'Vb'] or (pos in ['N', 'Nc' , 'Nb' ] and '_' in w)) ):
        if(pos in ['Np' ,  'Nb']  ):

            key_words_np.append(w)
    key_words_np = [word.lower() for word in key_words_np]
    key_words_np = list(set(key_words_np))
    key_words = extract_noun(filtered_combined_data)
    key_words = [word.lower() for word in key_words if word.count('_') >= 2]
    key_words.extend(key_words_np)
    key_words = list(set(key_words))

    key_words_dict[item['_id']] = {
        'keywords': key_words,
        'content': item['_source']['content']
    }
    # add_to_json_file('keyword_test_3.json', key_words_dict)

    

    # print(key_words) 
    
with open('keyword_test_3.json', 'w', encoding='utf-8') as file:
    json.dump(key_words_dict, file, ensure_ascii=False, indent=4)

    
 







