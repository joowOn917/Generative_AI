# Generative_AI

# Image Generator Project 


## 🖥 Overview 
사용자가 입력한 이미지와 텍스트 기반의 프롬포트를 활용하여 이미지를 생성하는 것이 목표이다. 

사용하는 모델은 StableDiffusion으로, 이는 자연어 설명으로부터 이미지를 생성해내는 전형적인 Text-to-Image task 모델이다. 

StableDiffusionXLImg2ImgPipeline 클래스를 활용하여 이미 학습된 StableDiffusion 모델을 로드하고, 입력 이미지를 받아 안정적인 방식으로 변환한 후 출력(결과)이미지를 생성할 것이다. 

## 🛠 Requirement 
- Python 3.11.5
- tensorflow 2.15
- scipy 1.11.1
- matplotlib 3.7.2
- pandas 2.0.3
- numpy 1.24.3
- torch 2.1.1
- torchvision 0.16.1
- opencv-python-headless
- fastapi 0.104.1
- konlpy 0.6.0


## ⚙ Project Process
<img src = "https://github.com/JinSan-RM/ImageGen_textPlusimage/assets/143769249/f1b63cd0-b5e8-4f39-92b7-30cb084125a7" width="80%" height="80%">


### 1️⃣ Preprocessing for Prompt
Text-to-Image에 사용하는 입력 텍스트를 프롬포트라 하며, 이 과정을 정확한 이미지 생성을 위해 입력 받은 텍스트를 전처리하는 과정이다. 

프롬포트는 한국어와 영어 두가지 버전으로 개발되었다.

#### 📜 영어 프롬포트
  **Step1.** 입력받은 텍스트와 유사한 문장 다수 생성 (Paraphrasing API) 
  
  **Step2.** 모든 문장들 영어로 번역 (Papago API)
  
  **Step3.** KeyBERT(자연어처리모델)를 활용해 키워드와 가중치 추출
  
  **Step4.** 텍스트에 가중치를 부여하여 프롬포트 형식으로 변환
  

#### 📜 한국어 프롬포트
  **Step1**. 입력 받은 텍스트와 유사한 문장 다수 생성 (Paraphrasing API)
  
  **Step2.** 모든 문장들 토큰화
  
  **Step3.** 조사 제거, 어간 추출
  
  **Step4.** 단어별 가중치 부여



## 📌 APIs & Model

* **Paraphrasing API**
  KT 지니랩스에서 제공하는 Open API

  입력한 문장의 구조 및 유사한 의미를 가진 문장을 AI를 통해 생성하여 데이터를 증강시키고, 데이터 셋을 빠르고 효과적으로 구축해준다.
    * input값: 문자열 리스트
    * diversity: 생성되는 문장의 유사도 ("1": 변화정도 적게, "2": 보통, "3": 크게)
    * domain: 요청할 서비스 분야 ("1": 발화, "2": 신문기사, "3": 기타)
    * data: 데이터 증강 요청할 문장 (최대 10개, 1~256자)


    [출처] https://genielabs.ai/tech/detail?domain=nlp&contentsSeq=114


+ **Papago API**

  네이버 개발자 센터에서 제공하는 Open API

  Papago의 인공 신경망 기반 기계 번역 기술(NMT, Neural Machine Translation)로 텍스트를 원하는 언어로 번역하여 반환해준다. 
  * input값: 문자열 
  * 원본 언어: Korean
  * 목적 언어: English


    [출처] https://genielabs.ai/tech/detail?domain=nlp&contentsSeq=114



* **KeyBERT**

  ***BERT**를 기반으로 한 모델로, 문서를 가장 잘 나타내는 키워드를 찾고 유사도를 추출해준다. 
  * keyphrase_ngram_range: 키워드/키 구문의 길이 
  * stop_words: 불용어


  <img src="https://github.com/JinSan-RM/ImageGen_textPlusimage/assets/143769249/d30fc8dd-27bd-46e4-8794-b781617479fd" width="80%" height="%80">
---
***BERT**: 자연어처리를 위해 2018년에 구글에서 고안한 transformers기반의 머신러닝 모델. 문장의 전체 구조를 양방향으로 학습하여 문맥을 파악한 뒤, 단어를 임베딩한다. 

