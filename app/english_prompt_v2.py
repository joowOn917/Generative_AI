# 영어 프롬포트 v2
# v1과 달리 bert를 사용하여 가중치를 부여하기에 토큰화 과정은 생략 
# prompt_weighting함수는 prompt에 가중치를 포함한 버전
# prompt 따로 prompt_2(키워드에 가중치 부여된 프롬포트) 따로 하는 경우에는 papago_translation과 positive_prompt만 사용

import requests
import pandas as pd
import json
from datetime import datetime
import hmac, hashlib
from pytz import timezone
from keybert import KeyBERT
from tensorflow.keras.preprocessing.text import text_to_word_sequence


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
papago_client_id = "fsnS2UC0g3Jw11PIynSc"  # 개발자 센터에서 발급받은 클라이언트 ID
papago_client_secret = 'xP1iJ3P8Ct'  # 개발자 센터에서 발급받은 클라이언트 시크릿
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
                text_list = result['result'][0]
                text_list.append(text)
                
                return text_list
            
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
            
            base_prompt = translated_texts[-1]
                
        return translated_texts, base_prompt
    
    # BERT모델을 활용한 키워드, 가중치 추출
    @classmethod
    def bert(cls, text_list):
        bow = []
        
        extractor = KeyBERT("all-mpnet-base-v2")
        
        for i in range(len(text_list)):
            keywords = extractor.extract_keywords(text_list[i], stop_words='english')
            bow.append(keywords)
            
        new_bow = []
        
        for i in range(0, len(bow)):
            for j in range(len(bow[i])):
                new_bow.append(bow[i][j])
                
        df = pd.DataFrame(new_bow, columns=['keyword', 'weight'])
        
        # GROUPBY
        # 동일 keyword에 대해서 가중치 중앙값 추출
        weight_df = df.groupby('keyword').agg('median').sort_index()
        # 동일 keyword의 count 추출 후, 칼럼명 count로 변환
        count_df = df.groupby('keyword').agg('count').sort_index()
        count_df.rename(columns={'weight':'count'}, inplace=True)
        
        # keyword기준 두 df 합치기
        result_df = weight_df.join(count_df)
        result_df.sort_values('count', ascending=False, inplace=True)
        
        return result_df
    
    @classmethod
    ## +로 가중치 부여한 프롬프트 
    def prompt_weighting(cls, dataframe, text_list):
        
        # 중요도 높은 단어 상위 5개 추출
        top_weight_word = dataframe.head(5).index.values
        
        weight_dict = {}
        prompt = ""
        
        # 단어:가중치 형태의 딕셔너리 생성
        for i in range(len(top_weight_word)):
            weight_dict[top_weight_word[i]] = '+' * (5-i)

        # 단어 옆에 가중치 추가
        for text in text_list:
            indexs = []
            for key, val in weight_dict.items():
                if key in text:
                    index = text.find(key)
                    
                    if index != -1:
                        text = text[:index + len(key)] + val + text[index + len(key):]
            prompt += text
        
        # 마침표 제거. 쉼표로 프롬포트 연결
        if "." in prompt:
            prompt.replace(".", ", ")
        
        return prompt[:-2]

    @classmethod 
    def positive_prompt(cls, df):
        result = ''
        keyword = df.index.values
        
        for i in range(len(keyword)):
            if i != len(keyword)-1:
                result = result + keyword[i] + '+++, '
            else:
                result = result + keyword[i] + '+++'
                
        return result
    
    
    