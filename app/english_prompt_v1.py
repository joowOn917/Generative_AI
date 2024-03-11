# 영어 프롬포트 v1
# 토큰화까지만 진행된 버전. 가중치 부여를 하지 못했기에 사용할일은 없음

import requests
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import json
from datetime import datetime
import hmac, hashlib
from pytz import timezone

# paraphrasing api 활용 변수
genie_client_id = 'glabs_faaf9d6e1c4be5e87ba79992eb5343787a458825fc17e9d143dadb3cc832115b'
genie_client_secret = 'd129ce0af3b0490eb57360783860bb589840c4ab4fa8af37b95c0339686ff8d0'
genie_client_key = "61dcc5e0-8836-59a6-946a-4bd9df293964"
genie_url = "https://aiapi.genielabs.ai/kt/nlp/paraphrasing"

timestamp = datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d%H%M%S%f")[:-3] 
signature = hmac.new(
    key = genie_client_secret.encode('UTF-8'), 
    msg = f"{genie_client_id}:{timestamp}".encode('UTF-8'), 
    digestmod = hashlib.sha256
).hexdigest()

genie_headers = {
"x-client-key":f"{genie_client_key}",
"x-client-signature":f"{signature}",
"x-auth-timestamp": f"{timestamp}",
"Content-Type": "application/json",
}

# papago api 활용 변수
papago_client_id = "w979m4b35v"  # 개발자 센터에서 발급받은 클라이언트 ID
papago_client_secret = 'lWh1jDMncHEAjJ2dlPkze5aauxNhqiVvRrPER507'  # 개발자 센터에서 발급받은 클라이언트 시크릿
papago_url = 'https://openapi.naver.com/v1/papago/n2mt'
papago_headers = {
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Naver-Client-Id': papago_client_id,
    'X-Naver-Client-Secret': papago_client_secret,
}


class PromptEnglish:
    def __init__(self, text):
        self.text = text
        
    def text_paraphrasing(self):
        body = json.dumps({"user_id" : "NIPA", 
                           "ret_sentences" : 50,
                           "diversity": 3, 
                           "domain": 2,
                           "data": (self.text, 256)
                           }) 

        response = requests.post(genie_url, data=body, headers=genie_headers)             # verify = False

        if response.status_code == 200:
            try:
                result = response.json()
                
                return result['result'][0]
            
            except json.decoder.JSONDecodeError:
                return f'Error decoding JSON: {str(e)}\n response.text: "{response.text}"'
        else:
            return f"HTTP error {response.status_code}\n{response.text}"

    @classmethod
    def papago_translation(cls, text_list):
        
        translated_texts = []
        
        # 한국어, 영어로 변환
        for text in text_list:
            data = {
                'source': 'ko',
                'target': 'en',
                'text': text,
            }

            response = requests.post(papago_url, headers=papago_headers, data=data)
            result = response.json()

            if 'errorMessage' in result:
                return result['errorMessage']

            else:
                translated_text = result['message']['result']['translatedText']
                translated_texts.append(translated_text)
                
        return translated_texts
    
    @classmethod
    def eng_tokenize(cls, trans_texts):
        # 불용어 말뭉치
        stop_words = stopwords.words('english')
        # 불용어 말뭉치에 포함되어 있는 부정어 말뭉치에서 제외
        survived = ['no', 'not', 'nor']
        
        for surv in survived:
            stop_words.remove(surv)
        # 만일을 대비한 중복제거
        stop_wrods = set(stop_words)
        
        result = []
        
        for text in trans_texts:
            text_token = text_to_word_sequence(text)
            temp = []
            
            for token in text_token:
                if token not in stop_words:
                    temp.append(token)
            
            result.append(temp)
            
        return result
    
