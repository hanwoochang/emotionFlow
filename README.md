Moive Emotion Analysis

 이 프로젝트는 영화 대본을 분석하여 등장인물의 감정 흐름과 기승전결의 분위기를 시각화하는 프로젝트입니다.

 snunlp/KR-Medium (BERT 계열) 모델을 기반으로 Fine-tuning 하였으며, 단순한 분류를 넘어 영화적 맥략을 고려한 감정 분석을 수행합니다.

 주요 기능
   6가지 핵심 감정(기쁨, 슬픔, 분노, 공포, 놀람, 중립)을 정밀하게 분류합니다

   기존 BERT 분류기 대신 커스텀 MLP Head(Linear -> ReLU -> Dropout -> Linear)를 적용하여 분류 성능을 강화하려고 하였습니다.

   단순 빈도수가 아닌 감정 강도 기반의 가중치 알고리즘 적용하여 기승전결 감정 흐름을 분석 및 시각화 하였습니다.

   데이터 불균형을 처리하기 위해 Class Weights와 Down-sampling을 적용하였습니다

<br><br>

사용 방벙
  1. 환경설정
     pip install torch transformers pandas matplotlib tqdm scikit-learn
  2. 모델학습
     python train.py
  3. 영화 분석
     python inference.py



<img width="1913" height="666" alt="스크린샷 2025-11-28 141302" src="https://github.com/user-attachments/assets/aa4f5650-8a46-4ed4-8d49-917b5cf0ce95" />



<img width="1898" height="666" alt="스크린샷 2025-11-28 175245" src="https://github.com/user-attachments/assets/5df6e776-54ac-4e46-a130-322356f59de1" />
