# 한국어 프롬포트를 위한 클래스 
# 토큰화와 키워드까지 추출했지만, 가중치 부여는 x

import requests
import json
from datetime import datetime
import hmac, hashlib
from pytz import timezone
from konlpy.tag import Okt
from collections import defaultdict 
from difflib import SequenceMatcher

client_id = 'glabs_faaf9d6e1c4be5e87ba79992eb5343787a458825fc17e9d143dadb3cc832115b'
client_secret = 'd129ce0af3b0490eb57360783860bb589840c4ab4fa8af37b95c0339686ff8d0'
client_key = "61dcc5e0-8836-59a6-946a-4bd9df293964"

timestamp = datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d%H%M%S%f")[:-3] 
signature = hmac.new(
    key = client_secret.encode('UTF-8'), 
    msg = f"{client_id}:{timestamp}".encode('UTF-8'), 
    digestmod = hashlib.sha256
).hexdigest()

headers = {
"x-client-key":f"{client_key}",
"x-client-signature":f"{signature}",
"x-auth-timestamp": f"{timestamp}",
"Content-Type": "application/json",
}

url = "https://aiapi.genielabs.ai/kt/nlp/paraphrasing"

class KoreanPrompt:
    def __init__(self, text):
        self.text = text
        
    def get_response(self):
        body = json.dumps({"user_id" : "NIPA", 
                           "ret_sentences" : 50,
                           "diversity": 3, 
                           "domain": 2,
                           "data": (self.text, 256)
                           }) 

        response = requests.post(url, data=body, headers=headers)             # verify = False

        if response.status_code == 200:
            try:
                # 유사문장 리스트 (결과값)
                result = response.json()
                return result['result'][0]
            
            except json.decoder.JSONDecodeError:
                return f'Error decoding JSON: {str(e)}\n response.text: "{response.text}"'
        else:
            return f"HTTP error {response.status_code}\n{response.text}"
    
    @classmethod    
    def get_tokens(cls, text_list):
        okt = Okt()
        # print(text_list)

        # 딕셔너리 형태로 전처리 된 모든 유사 문장들 (리스트)
        result_dict = []
        
        for text in text_list:
            # 단순 형태소 분리
            pos = okt.pos(text)
            # 어간 추출, 형태소 분리
            stem_pos = okt.pos(text, stem=True)
            
            # 불용어 제거
            pos_tag = [p for p in pos if (p[1] == 'Noun' or p[1] == 'Verb' or p[1] == 'Adjective' or (p[1] == 'Punctuation' and p[0] == '!'))] 
            stem_pos_tag = [s for s in stem_pos if (s[1] == 'Noun' or s[1] == 'Verb' or s[1] == 'Adjective' or (s[1] == 'Punctuation' and s[0] == '!'))] 

            # {원형: 변형} 형태의 딕셔너리
            # ex) {먹다: [먹는, 먹고]}
            morph_stem_pos = {}
        
            for i in range(len(pos_tag)):
                key = stem_pos_tag[i][0]
                val = pos_tag[i][0]
                morph_stem_pos[key] = val
                # print(morph_stem_pos)
                
            result_dict.append(morph_stem_pos)
                
        return result_dict 
    
    @classmethod
    def weight_words(cls, text_pos_dict):
        result = defaultdict(list)
        
        for text_dict in text_pos_dict:
            for key, val in text_dict.items():
                result[key].append(val)
                
        result = dict(result)
        
        for key, values in result.items():
            # 유사 단어
            similar_word = []
            # 유사도 
            similarity = []
            
            # 단어 유사도 측정
            for value in values:
                similar = SequenceMatcher(None, key, value).ratio()
                similarity.append(round(similar,3))
                
            similar_word.append(values)
            similar_word.append(similarity)
            
            result[key] = similar_word
            
        return result

