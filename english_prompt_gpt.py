# 영어 프롬포트 v3
# 지니랩스의 paraphrasing api가 아닌 gpt api 사용

import requests
import pandas as pd
import json
from datetime import datetime
import hmac, hashlib
from pytz import timezone
from keybert import KeyBERT
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import openai

# gpt api 관련 변수
openai.api_key = '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'

# papago api 관련 변수
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
        
    # gpt api를 활용해 입력값과 유사한 문장 10개 생성
    def gpt(self):
        input_message = f'"{self.text}"와 유사한 문장 10개만 생성해줘. 디테일한 내용이 더 추가되면 더 좋아!'
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": "You are helpful assistant."},
                    {"role": "user", "content": input_message}
                ],
                temperature=0
            )
            improved_texts = response.choices[0].message.content

            # 하나의 문자열로 들어오는 10개의 문장들을 분리하고 리스트에 담기 (1. 문장1, 2. 문장2, ...)
            sentences = [text.strip() for text in improved_texts.split('\n')]
                    
            # 증강된 문장들 담을 리스트
            augments = []
            # 1, 2로 매겨져 있는 문장번호 제거하고 한문장씩 리스트에 담기
            for sentence in sentences:
                temp = sentence.split('.')
                hap = '' 
                for i in range(len(temp)):
                    if i == 0:
                        pass
                    elif i == len(temp)-1:
                        hap += temp[i] 
                    else:
                        hap = hap + temp[i] + '.'
                augments.append(hap)

            # 초기 입력값도 리스트에 추가
            augments.append(self.text)
            return augments
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return str(e)

    # papago api를 활용한 모든 문장들 번역
    @classmethod
    def papago_translation(cls, text_list):
        translated_texts = []       # 번역된 문장들 담을 리스트
        
        # 한국어 -> 영어로 변환
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
            
            # 기본 프롬프트(prompt)가 될 입력값 번역 버전
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

    # 추출한 키워드들만 모아서 가중치 부여하고 prompt_2로 사용
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
    
    ## 모든 문장들에 가중치 부여
    # 하나의 프롬포트로 할 경우 사용 (사용 안함)
    @classmethod 
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