# common.py
import os
import platform
import json
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt

# 1. 설정 (공통)
CONFIG = {
    'model_name': 'snunlp/KR-Medium',
    'max_length': 128,
    'batch_size': 16,
    'epochs': 5,
    'learning_rate': 2e-5,
    'excluded_emotions': ['미분류'], 
    'output_dir': './results',
    # 추론 관련 설정
    'analyze_movie_path': './test/01_1000.json',
    'analysis_batch_size': 32,
    'shift_window_size': 10,
    'shift_threshold': 0.5,
}

EMOTION_MAPPING = {
    # 1. 기쁨 (20개)
    '기쁘다': '기쁨', '행복하다': '기쁨', '즐겁다': '기쁨', '만족하다': '기쁨',
    '반갑다': '기쁨', '고맙다': '기쁨', '감동하다': '기쁨', '감탄하다': '기쁨',
    '사랑스럽다': '기쁨', '설레다': '기쁨', '기대되다': '기쁨', '뿌듯하다': '기쁨',
    '자랑스럽다': '기쁨', '자신감 있다': '기쁨', '안심/안도하다': '기쁨',
    '애틋하다': '기쁨', '상쾌하다': '기쁨', '통쾌하다': '기쁨', '열광하다': '기쁨',
    '황홀하다': '기쁨',

    # 2. 슬픔 (15개)
    '슬프다': '슬픔', '우울하다': '슬픔', '서럽다': '슬픔', '비참하다': '슬픔',
    '괴롭다': '슬픔', '절망하다': '슬픔', '안타깝다': '슬픔', '속상하다': '슬픔',
    '서운하다': '슬픔', '씁쓸하다': '슬픔', '미안하다': '슬픔', '주눅들다': '슬픔',
    '창피하다': '슬픔', '부끄럽다': '슬픔', '피곤하다': '슬픔',

    # 3. 분노 (9개)
    '화나다': '분노', '짜증나다': '분노', '증오하다': '분노',
    '불만족스럽다': '분노', '불쾌하다': '분노', '배신감': '분노',
    '약오르다': '분노', '답답하다': '분노', '거부감': '분노',

    # 4. 공포 (6개)
    '무섭다': '공포', '불안하다': '공포', '초조하다': '공포',
    '긴장되다': '공포', '걱정': '공포', '절박하다': '공포',

    # 5. 놀람 (5개)
    '놀라다': '놀람', '당황하다': '놀람', '황당하다': '놀람',
    '난감하다': '놀람', '심란하다': '놀람',

    # 6. 혐오 (3개)
    '뻔뻔하다': '혐오', '귀찮다': '혐오', '불편하다': '혐오',

    # 7. 중립 (7개)
    '담담하다': '중립', '단호하다': '중립', '비장하다': '중립',
    '궁금하다': '중립', '의심스럽다': '중립', '망설이다': '중립',
    '부럽다': '중립'
}

emotion_score_map = {
    '기쁨': 1.0,
    '중립': 0.0,
    '놀람': -0.1,
    '혐오': -0.6,
    '슬픔': -0.8,
    '분노': -0.9,
    '공포': -1.0
}

# 3. 모델 클래스 (학습/추론에서 동일하게 사용)
class KRMediumWithMLP(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.num_labels = num_labels
        self.config = self.base_model.config
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.hidden_size = self.config.hidden_size
        
        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_labels)
        )
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.base_model(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids if token_type_ids is not None else None,
              return_dict=True
        )
        pooled_output = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.mlp_head(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return (loss, logits) if loss is not None else (logits,)

# 4. 데이터셋 클래스
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding='max_length',
            max_length=max_length, return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

# 5. 유틸리티: 한글 폰트 설정
def set_korean_font():
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Darwin':
        plt.rc('font', family='AppleGothic')
    else:
        try:
            import matplotlib.font_manager as fm
            font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
                plt.rc('font', family='NanumGothic')
            else:
                pass
        except Exception:
            plt.rc('font', family='DejaVu Sans') 
    plt.rcParams['axes.unicode_minus'] = False