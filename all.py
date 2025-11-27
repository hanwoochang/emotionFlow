import os, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer, AutoModel, 
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 한글 폰트 설정
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

# 설정 (Test Mode 관련 항목 제거됨)
CONFIG = {
    'model_name': 'snunlp/KR-Medium',
    'max_length': 128,
    'batch_size': 16,
    'epochs': 5,
    'learning_rate': 5e-5,
    'excluded_emotions': ['미분류'], 
    'output_dir': Path('./results'),
    'analyze_movie_path': './test/1121.json',
    'analysis_batch_size': 32,
    'shift_window_size': 10, # 감정 전환점 분석을 위한 윈도우 크기
    'shift_threshold': 0.5,  # 감정 전환점 판단 임계값 (이전 대비 변화량)
}

# 7대 대표 감정으로 변환하는 매핑 테이블
EMOTION_MAPPING = {
    # 1. 기쁨 (Positive)
    '기쁨': '기쁨', '행복': '기쁨', '설렘': '기쁨', '만족': '기쁨', 
    '안정': '기쁨', '고마움': '기쁨', '기대감': '기쁨', '느긋': '기쁨', 
    '사랑': '기쁨', '자신감 있다': '기쁨', '재미': '기쁨', '즐거움': '기쁨', 
    '편안함': '기쁨', '신뢰': '기쁨',
    
    # 2. 슬픔 (Sadness)
    '슬픔': '슬픔', '우울': '슬픔', '상실감': '슬픔', '괴로움': '슬픔', 
    '외로움': '슬픔', '힘듦': '슬픔', '지침': '슬픔', '피곤': '슬픔', 
    '실망': '슬픔', '서운하다': '슬픔', '미안': '슬픔', '후회하다': '슬픔', 
    '부끄러움': '슬픔', '애처로운': '슬픔', '심각': '슬픔', '허탈함': '슬픔',
    
    # 3. 분노 (Anger)
    '분노': '분노', '화나다': '분노', '짜증': '분노', '불만': '분노', 
    '억울함': '분노', '질투': '분노',
    
    # 4. 공포 (Fear/Anxiety)
    '공포': '공포', '두려움': '공포', '불안': '공포', '걱정': '공포', 
    '초조': '공포', '의심': '공포', '혼란': '공포', '조급함': '공포',
    
    # 5. 놀람 (Surprise)
    '놀람': '놀람', '당황': '놀람', '황당': '놀람', '어이없음': '놀람',
    
    # 6. 혐오 (Disgust)
    '혐오': '혐오', '불쾌': '혐오', '싫증': '혐오',
    
    # 7. 중립 (Neutral)
    '중립': '중립', '담담하다': '중립', '무관심': '중립', '지루함': '중립', 
    '신중': '중립', '궁금함': '중립'
}

# 새로운 7개 감정의 점수 맵
emotion_score_map = {
    '기쁨': 1.0,
    '중립': 0.0,
    '놀람': -0.1, 
    '슬픔': -0.8,
    '분노': -0.9,
    '공포': -1.0, 
    '혐오': -0.7
}

# 커스텀 모델: KR-Medium + MLP 분류기
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
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        logits = self.mlp_head(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return (loss, logits) if loss is not None else (logits,)

def load_emotion_data(folder_path, excluded_emotions=None):
    folder = Path(folder_path)
    json_files = list(folder.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f"'{folder_path}'에 JSON 파일이 없습니다.")
    
    texts, emotions = [], []
    skipped_count = 0
    
    for file_path in tqdm(json_files, desc=f"로딩 {folder_path}"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for unit in data.get('units', []):
                for script in unit.get('story_scripts', []):
                    content = script.get('content')
                    raw_emotion = script.get('emotion') # 원본 감정
                    
                    if isinstance(raw_emotion, list):
                        raw_emotion = raw_emotion[0] if raw_emotion else None
                        
                    if content and raw_emotion:
                        # 원본 감정을 7대 감정으로 변환
                        mapped_emotion = EMOTION_MAPPING.get(raw_emotion)
                        
                        # 매핑된 감정이 있고, 제외 목록에 없으면 추가
                        if mapped_emotion and mapped_emotion not in (excluded_emotions or []):
                            texts.append(content)
                            emotions.append(mapped_emotion) # 변환된 라벨 저장
                        else:
                            skipped_count += 1
                            
        except Exception as e:
            print(f"⚠️ {file_path.name} 읽기 실패: {e}")
    
    print(f"✓ 총 {len(texts)}개 샘플 로드 (7대 감정 변환 완료), {skipped_count}개 제외됨")
    return texts, emotions

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding='max_length',
            max_length=max_length, 
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # 정밀도, 재현율, F1 점수 계산 (weighted: 클래스 불균형 고려)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_emotion_classifier():
    print("\n🚀 감정 분류 모델 학습 시작 (MLP 분류기 포함)\n")
    
    X_train, y_train_text = load_emotion_data('./train_data/label/', CONFIG['excluded_emotions'])
    X_val, y_val_text = load_emotion_data('./validation_data/label/', CONFIG['excluded_emotions'])

    valid_emotions = set(y_train_text)
    X_val_filtered = [x for x, y in zip(X_val, y_val_text) if y in valid_emotions]
    y_val_filtered = [y for y in y_val_text if y in valid_emotions]
    
    # Validation 데이터가 비었을 경우의 안전장치
    if not X_val_filtered:
        X_val_filtered, y_val_filtered = X_train[:100], y_train_text[:100]
    
    print(f"📊 Train 샘플: {len(X_train)}, Validation 샘플: {len(X_val_filtered)}")
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_val = label_encoder.transform(y_val_filtered)
    num_labels = len(label_encoder.classes_)
    
    emotion_dist = pd.Series(y_train_text).value_counts()
    print(f"\n📈 Train 감정 분포 :")
    for emotion, count in emotion_dist.head(10).items():
        print(f"   {emotion}: {count}개")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = KRMediumWithMLP(
        model_name=CONFIG['model_name'], 
        num_labels=num_labels
    )

    train_dataset = EmotionDataset(X_train, y_train, tokenizer, CONFIG['max_length'])
    val_dataset = EmotionDataset(X_val_filtered, y_val, tokenizer, CONFIG['max_length'])

    training_args = TrainingArguments(
        output_dir=str(CONFIG['output_dir']),
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    print("\n🔥 학습 시작...")
    trainer.train()

    plot_training_history(trainer.state.log_history, CONFIG['output_dir'])

    output_path = CONFIG['output_dir'] / 'final_model_mlp'
    output_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_path / 'pytorch_model.bin')
    model.config.to_json_file(output_path / 'config.json')
    tokenizer.save_pretrained(output_path)
    
    with open(output_path / 'label_encoder.json', 'w', encoding='utf-8') as f:
        json.dump({'classes': label_encoder.classes_.tolist()}, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 모델 저장 완료: {output_path}")
    
    eval_result = trainer.evaluate()
    print(f"\n📊 최종 Validation Accuracy: {eval_result['eval_accuracy']:.4f}")
    
    return output_path

def load_model_and_predict(movie_path, model_dir):
    print(f"\n🎬 영화 감정 분석: {movie_path}")
    
    with open(model_dir / 'label_encoder.json', 'r', encoding='utf-8') as f:
        label_data = json.load(f)
        id2label = {i: label for i, label in enumerate(label_data['classes'])}
        num_labels = len(label_data['classes'])
        
    model = KRMediumWithMLP(CONFIG['model_name'], num_labels)
    model.load_state_dict(torch.load(model_dir / 'pytorch_model.bin'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with open(movie_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = data.get('text', [])
    
    if not texts:
        raise ValueError("영화 파일에 'text' 필드가 없거나 비어있습니다.")
    
    print(f"📝 총 {len(texts)}개 장면 분석 중...")
    
    pred_dataset = EmotionDataset(texts, None, tokenizer, CONFIG['max_length'])
    pred_dataloader = DataLoader(pred_dataset, batch_size=CONFIG['analysis_batch_size'])
    
    all_pred_ids = []
    
    with torch.no_grad():
        for batch in tqdm(pred_dataloader, desc="예측"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs) 
            logits = outputs[0]
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            all_pred_ids.extend(pred_ids)

    pred_emotions = [id2label[i] for i in all_pred_ids]
    
    emotion_counts = pd.Series(pred_emotions).value_counts()
    print(f"\n📊 예측된 감정 분포 (상위 10개):")
    for emotion, count in emotion_counts.head(10).items():
        print(f"   {emotion}: {count}개 ({count/len(pred_emotions)*100:.1f}%)")
    
    return texts, pred_emotions


def analyze_emotion_shifts(predictions, texts, window_size=10, threshold=0.8):
    """감정 점수의 급격한 변화(전환점)를 분석"""
    print(f"\n🔄 감정 전환점 분석 시작 (윈도우 크기: {window_size}, 임계값: {threshold})")
    
    scores = np.array([emotion_score_map.get(e, 0) for e in predictions])
    n = len(scores)
    
    shift_points = []
    
    # 이동 평균 계산 (감정 노이즈 제거)
    smoothed_scores = pd.Series(scores).rolling(window=window_size, min_periods=1, center=False).mean().values
    
    # 전환점 찾기
    for i in range(window_size, n - window_size):
        # 윈도우 크기 이전의 평균
        prev_avg = np.mean(smoothed_scores[i - window_size:i])
        # 윈도우 크기 이후의 평균
        next_avg = np.mean(smoothed_scores[i:i + window_size])
        
        # 감정 변화량 (절대값 변화)
        change_magnitude = abs(next_avg - prev_avg)
        
        # 긍정/부정 극성이 바뀌었는지 확인 (예: 긍정 -> 부정, 또는 부정 -> 긍정)
        # 이전 평균이 중립(0) 근처가 아니면서 극성이 바뀌고, 변화량이 임계값 이상인 경우
        if (prev_avg * next_avg < 0 and change_magnitude >= threshold):
            
            shift_type = "긍정 ➡️ 부정" if prev_avg > next_avg else "부정 ➡️ 긍정"
            
            # 이미 가까운 지점(window_size/2 이내)에 전환점이 기록되었다면 건너뜀
            is_new_shift = True
            for existing_point in shift_points:
                if abs(existing_point['index'] - i) < window_size // 2:
                    is_new_shift = False
                    break
            
            if is_new_shift:
                shift_points.append({
                    'index': i,
                    'type': shift_type,
                    'prev_avg': f"{prev_avg:.2f}",
                    'next_avg': f"{next_avg:.2f}",
                    'context': texts[i]
                })

    if shift_points:
        print(f"✅ 총 {len(shift_points)}개의 주요 감정 전환점 발견:")
        for point in shift_points:
            print(f"   * 장면 {point['index'] + 1} ({point['type']}): 감정 변화 ({point['prev_avg']} -> {point['next_avg']})")
            print(f"     > 주요 문맥: '{point['context'][:30]}...'")
    else:
        print("💡 주요 감정 전환점(임계값 이상)이 발견되지 않았습니다.")
        
    return shift_points

def plot_training_history(history, output_dir):
    """학습 로그를 바탕으로 Loss와 성능 지표 그래프를 그립니다."""
    print("\n📊 학습 결과 시각화 중...")
    
    train_loss = []
    train_steps = []
    eval_loss = []
    eval_steps = []
    
    eval_acc = []
    eval_prec = []
    eval_f1 = []
    
    # 로그 데이터 분리
    for log in history:
        if 'loss' in log:  # 학습 손실
            train_loss.append(log['loss'])
            train_steps.append(log['step'])
        if 'eval_loss' in log:  # 검증 손실 및 지표
            eval_loss.append(log['eval_loss'])
            eval_steps.append(log['step'])
            # 지표가 있으면 저장
            if 'eval_accuracy' in log: eval_acc.append(log['eval_accuracy'])
            if 'eval_precision' in log: eval_prec.append(log['eval_precision'])
            if 'eval_f1' in log: eval_f1.append(log['eval_f1'])
            
    plt.figure(figsize=(14, 6))
    
    # 1. Loss 그래프 (좌측)
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_loss, label='Train Loss', alpha=0.6, color='salmon')
    plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='o', color='steelblue')
    plt.title('Loss 변화 (손실값)', fontsize=14)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Metrics 그래프 (우측)
    plt.subplot(1, 2, 2)
    if eval_acc: plt.plot(eval_steps, eval_acc, label='Accuracy (정확도)', marker='s')
    if eval_prec: plt.plot(eval_steps, eval_prec, label='Precision (정밀도)', marker='^', linestyle='--')
    if eval_f1: plt.plot(eval_steps, eval_f1, label='F1 Score', marker='x', linestyle=':')
    
    plt.title('성능 지표 변화', fontsize=14)
    plt.xlabel('Steps')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)  # 0~1 사이로 고정
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / 'training_history.png'
    plt.savefig(save_path, dpi=150)
    print(f"💾 학습 그래프 저장 완료: {save_path}")
    plt.show()


def plot_emotion_flow(predictions, title, mode='segment'):
    """
    감정 흐름 시각화
    """
    scores = [emotion_score_map.get(e, 0) for e in predictions]
    
    if mode == 'smooth':
        window_size = max(10, len(scores) // 20)
        smoothed = pd.Series(scores).rolling(window=window_size, center=True).mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(smoothed, color='steelblue', linewidth=2, label='감정 흐름')
        plt.fill_between(range(len(smoothed)), smoothed, alpha=0.3, color='steelblue')
        plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        
        n = len(scores)
        plt.axvline(n//3, color='gray', linestyle=':', alpha=0.5)
        plt.axvline(2*n//3, color='gray', linestyle=':', alpha=0.5)
        plt.title(f"'{title}' 감정 흐름 (이동평균)", fontsize=17, pad=20)
        plt.xlabel("장면 순서", fontsize=13)
        plt.ylabel("감정 점수 (-1 ~ +1)", fontsize=13)
        plt.legend()
        
    else:
        n = len(predictions)
        
        indices_list = [
            (0, 1),                             # 1. 처음 (딱 첫 번째 문장)
            (0, int(n * 0.25)),                 # 2. 기 (0~25%)
            (int(n * 0.25), int(n * 0.5)),      # 3. 승 (25~50%)
            (int(n * 0.5), int(n * 0.75)),      # 4. 전 (50~75%)
            (int(n * 0.75), n),                 # 5. 결 (75~100%)
            (n - 1, n)                          # 6. 후반 (딱 마지막 문장)
        ]
        
        labels = ['처음(Start)', '기', '승', '전', '결', '마지막(End)']
        
        parts = [predictions[start:end] for start, end in indices_list]

        def segment_stats(preds):
            """가중치(빈도 * 강도)가 가장 높은 감정을 대표 감정으로 선정"""
            if not preds: return 0, "없음"
            
            counts = pd.Series(preds).value_counts()
            names = counts.index
            frequencies = counts.values
            
            seg_scores = np.array([emotion_score_map.get(n, 0) for n in names])
            
            weights = frequencies * np.abs(seg_scores) + 1e-6
            weighted_avg = np.sum(seg_scores * weights) / np.sum(weights)
            
            max_weight_idx = np.argmax(weights)
            dominant = names[max_weight_idx]
            
            return weighted_avg, dominant
        
        segments, averages, dominants = [], [], []
        
        for name, part in zip(labels, parts):
            if len(part) > 0: 
                avg, dom = segment_stats(part)
                segments.append(name)
                averages.append(avg)
                dominants.append(dom)
        
        plt.figure(figsize=(12, 6))
        plt.plot(segments, averages, color='gray', linewidth=2, alpha=0.7, marker='o', markersize=8)
        
        for x, y, e in zip(segments, averages, dominants):
            color = 'salmon' if y < 0 else 'cornflowerblue'
            size = 22 if x in ['처음(Start)', '마지막(End)'] else 15
            
            plt.plot(x, y, 'o', color=color, markersize=size)
            
            v_offset = 0.1 if segments.index(x) % 2 == 0 else -0.15
            if y > 0.8: v_offset = -0.15 
            if y < -0.8: v_offset = 0.1
            
            plt.text(x, y + v_offset, f"{e}\n({y:.2f})", ha='center', fontsize=11, fontweight='bold', color='black')
        
        plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        plt.title(f"'{title}' 상세 감정 흐름 (첫장면 vs 기승전결 vs 끝장면)", fontsize=17, pad=20)
        plt.ylabel("감정 점수", fontsize=13)
        plt.ylim(-1.2, 1.2)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = f"{title}_emotion_flow_{mode}_start_end.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n💾 상세 그래프 저장: {output_file}")
    plt.show()

def plot_both_flows(predictions, title, texts):
    """구간별 + 이동평균 두 가지 모두 생성 및 전환점 분석 실행"""
    print("\n📊 감정 흐름 시각화 중...")
    
    # 1. 문맥 기반 가중치 구간별 분석
    plot_emotion_flow(predictions, title, mode='segment')
    
    # 2. 이동평균 시각화
    plot_emotion_flow(predictions, title, mode='smooth')
    
    # 3. 감정 전환점 분석 실행
    analyze_emotion_shifts(
        predictions, 
        texts, 
        window_size=CONFIG['shift_window_size'], 
        threshold=CONFIG['shift_threshold']
    )

if __name__ == "__main__":
    try:
        # 학습
        model_path = train_emotion_classifier()
        
        # 분석
        if model_path and Path(CONFIG['analyze_movie_path']).exists():
            texts, preds = load_model_and_predict(CONFIG['analyze_movie_path'], model_path)
            movie_title = Path(CONFIG['analyze_movie_path']).stem
            plot_both_flows(preds, movie_title, texts)
        else:
            print(f"⚠️ 분석할 영화 파일이 없습니다: {CONFIG['analyze_movie_path']}")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()