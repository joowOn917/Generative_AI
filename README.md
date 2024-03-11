
# Image Generator Project 
사용자가 입력한 텍스트 기반의 프롬포트를 활용하여 이미지를 생성하는 것이 목표.

## 🖥 Overview 
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

StableDiffusion에는 총 3가지의 prompt 관련 하이퍼 파라미터가 존재한다. 

1. prompt: 입력받은 프롬포트
2. prompt_2: 이미지 생성에 영향을 주는 설명 관련 키워드들과 퀄리티 관련 긍정적인 키워드들 (퀄리티 관련 단어들은 default)
3. negative_prompt: 이미지 생성에 영향을 주는 퀄리티 관련 부정적인 키워드들 (default)


### 1️⃣ Preprocessing for Prompt

사용자로부터 한국어로 input prompt를 받고, 이를 영어로 변환하여 StableDiffusion모델에 활용할 최종 prompt와 prompt_2를 생성하는 과정이다. 

  **Step1.** 입력받은 한국어 텍스트와 유사한 문장 다수 생성 (Paraphrasing API or GPT API) 
  
  **Step2.** 모든 문장들 영어로 번역 (Papago API)

  **Step3.** 번역된 입력값은 'prompt'로 지정
  
  **Step4.** KeyBERT(자연어처리모델)를 활용해 증강된 문장들에서 자주 등장하는 키워드 추출
  
  **Step5.** 추출된 키워드들에 빈도수를 활용하여 가중치 부여 후, 이를 'prompt_2'로 지정
 

### 2️⃣ Tunning Model & Generating Image

학습된 StableDiffusion 모델을 사용하면서, 원하는 최적을 결과물을 위해 일부 파라미터들을 튜닝한다. 

**Step 1.** 모델 불러오기

```
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
)
pipe.to('cuda')
```
**Step 2.** 하이퍼 파라미터 정의

```
# 퀄리티 관련 default prompts (pos and neg)
positive_prompt = 'realistic photography+++, realism++, realistic++, super-detailed++, true-to-life++, best quality++, balanced++, well-defined++, clear++, well-defined facial features++, detailed faces++'
negative_prompt = 'CG, wallpaper, animation, anime, doll, disney, cartoons, cropped, misshapen, blurry, unfocused, desaturated, abstract, surreal, pixelated, noisy, pop art, no faces, no objects, no landscape, mutilated, disfigured, ugly, deformed, clear visibility of faces, no abstract faces, shadows on faces'

prompt_2 += positive_prompt

# num_inference_steps : 추론 단계의 수 (default=50)
n = 130
# guidance_scale : 프롬프트에 비슷한 정도 (보통 7~15)
m = 7.5
```

**Step3.** 이미지 생성  
```
image = pipe(prompt=prompt, prompt_2=prompt_2, negative_prompt=negative_prompt, num_inference_steps=n, guidance_scale=m).images[0]
```


## 사용한 APIs & Model

* **Paraphrasing API**
  
  KT 지니랩스에서 제공하는 Open API

  입력한 문장의 구조 및 유사한 의미를 가진 문장을 AI를 통해 생성하여 데이터를 증강시키고, 데이터 셋을 빠르고 효과적으로 구축해준다.
  
  [출처] https://genielabs.ai/tech/detail?domain=nlp&contentsSeq=114

* **GPT API**

   Openai에서 제공하는 GPT API 

   GPT에서 보낸 message의 답변을 활용한다. 

+ **Papago API**

  네이버 개발자 센터에서 제공하는 Open API

  Papago의 인공 신경망 기반 기계 번역 기술(NMT, Neural Machine Translation)로 텍스트를 원하는 언어로 번역하여 반환해준다. 

  [출처] https://genielabs.ai/tech/detail?domain=nlp&contentsSeq=114



* **KeyBERT**

  ***BERT**를 기반으로 한 모델로, 문서를 가장 잘 나타내는 키워드를 찾고 유사도를 추출해준다. 
  * keyphrase_ngram_range: 키워드/키 구문의 길이 
  * stop_words: 불용어


  <img src="https://github.com/JinSan-RM/ImageGen_textPlusimage/assets/143769249/d30fc8dd-27bd-46e4-8794-b781617479fd" width="80%" height="%80">


---
***BERT**: 자연어처리를 위해 2018년에 구글에서 고안한 transformers기반의 머신러닝 모델. 문장의 전체 구조를 양방향으로 학습하여 문맥을 파악한 뒤, 단어를 임베딩한다. 

