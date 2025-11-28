#train.py
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix 
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 공통 모듈 임포트
from common import CONFIG, EMOTION_MAPPING, KRMediumWithMLP, EmotionDataset, set_korean_font

# 데이터 로드 함수 
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
                    raw_emotion = script.get('emotion')
                    
                    if isinstance(raw_emotion, list):
                        raw_emotion = raw_emotion[0] if raw_emotion else None
                        
                    if content and raw_emotion:
                        mapped_emotion = EMOTION_MAPPING.get(raw_emotion)
                        if mapped_emotion and mapped_emotion not in (excluded_emotions or []):
                            texts.append(content)
                            emotions.append(mapped_emotion)
                        else:
                            skipped_count += 1
        except Exception as e:
            print(f"⚠️ {file_path.name} 읽기 실패: {e}")
    
    print(f"✓ 총 {len(texts)}개 샘플 로드, {skipped_count}개 제외됨")
    return texts, emotions

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# 학습 그래프 + 혼동 행렬을 함께 그리는 함수
def plot_training_results(history, output_dir, y_true, y_pred, classes):
    set_korean_font()
    print("\n📊 통합 결과 시각화 중 (Loss, Score, Matrix)...")
    
    train_loss = [x['loss'] for x in history if 'loss' in x]
    train_steps = [x['step'] for x in history if 'loss' in x]
    
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    eval_steps = [x['step'] for x in history if 'eval_loss' in x]
    eval_acc = [x['eval_accuracy'] for x in history if 'eval_accuracy' in x]
    eval_f1 = [x['eval_f1'] for x in history if 'eval_f1' in x]  

    # 1행 3열 그래프 생성
    fig = plt.figure(figsize=(20, 6))
    
    # [1] Loss 변화
    plt.subplot(1, 3, 1)
    plt.plot(train_steps, train_loss, label='Train Loss', alpha=0.6, color='salmon')
    plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='o', color='steelblue')
    plt.title('Loss 변화 (낮을수록 좋음)')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # [2] 성능 지표 (Accuracy & F1)
    plt.subplot(1, 3, 2)
    if eval_acc: 
        plt.plot(eval_steps, eval_acc, label='Accuracy', marker='s', color='steelblue', linestyle='--')
    if eval_f1: 
        plt.plot(eval_steps, eval_f1, label='F1 Score', marker='^', color='darkorange', linewidth=2)
    plt.title('성능 지표 변화 (높을수록 좋음)')
    plt.xlabel('Steps')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # [3] 혼동 행렬 (Confusion Matrix)
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_true, y_pred)
    # 정규화 (비율로 표시)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.title('최종 검증 혼동 행렬 (Confusion Matrix)')
    plt.xlabel('예측값 (Predicted)')
    plt.ylabel('실제값 (Actual)')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_result_summary.png')
    print(f"💾 결과 그래프 저장 완료: {output_dir / 'training_result_summary.png'}")
    plt.show()

def balance_dataset(texts, labels, target_count=40000):
    df = pd.DataFrame({'text': texts, 'label': labels})
    
    balanced_dfs = []
    
    for label_class in df['label'].unique():
        sub_df = df[df['label'] == label_class]
        count = len(sub_df)
        
        if count > target_count:
            sampled = sub_df.sample(n=target_count, random_state=42)
            balanced_dfs.append(sampled)
        else:
            balanced_dfs.append(sub_df)
            
    final_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n⚖️ 데이터 밸런싱 완료: 각 클래스 최대 {target_count}개로 조정됨")
    print(final_df['label'].value_counts())
    
    return final_df['text'].tolist(), final_df['label'].tolist()

def main():
    print("\n🚀 감정 분류 모델 학습 시작 \n")
    
    # 1. 데이터 로드
    print("Step 1. 데이터 로드 중...")
    X_train_raw, y_train_text_raw = load_emotion_data('./train_data/label/', CONFIG['excluded_emotions'])
    X_val, y_val_text = load_emotion_data('./validation_data/label/', CONFIG['excluded_emotions'])

    # 2. 데이터 밸런싱
    print("Step 2. 데이터 밸런싱 적용 중...")
    X_train, y_train_text = balance_dataset(X_train_raw, y_train_text_raw, target_count=40000)

    # 3. 라벨 인코딩
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    
    # Validation 필터링 
    valid_classes = set(label_encoder.classes_)
    X_val_filtered = []
    y_val_filtered_text = []
    for x, y in zip(X_val, y_val_text):
        if y in valid_classes:
            X_val_filtered.append(x)
            y_val_filtered_text.append(y)
            
    y_val = label_encoder.transform(y_val_filtered_text)
    num_labels = len(label_encoder.classes_)
    
    print(f"🏷️ 분류 클래스: {label_encoder.classes_}")

    # 4. 클래스 가중치 계산
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    print(f"⚖️ 클래스 가중치: {class_weights}")

    # 5. 모델 초기화 
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = KRMediumWithMLP(
        model_name=CONFIG['model_name'], 
        num_labels=num_labels,
        class_weights=class_weights 
    )

    # 6. 데이터셋 생성
    train_dataset = EmotionDataset(X_train, y_train, tokenizer, CONFIG['max_length'])
    val_dataset = EmotionDataset(X_val_filtered, y_val, tokenizer, CONFIG['max_length'])

    # 7. 학습 설정
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=0.1,        
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
    
    # 학습 시작
    trainer.train()

    print("\n🏁 학습 완료! 최종 검증 데이터 평가 중...")
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    plot_training_results(
        trainer.state.log_history, 
        CONFIG['output_dir'],
        y_true, 
        y_pred, 
        label_encoder.classes_
    )

    output_path = Path(CONFIG['output_dir']) / 'final_model_mlp'
    output_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_path / 'pytorch_model.bin')
    model.config.to_json_file(output_path / 'config.json')
    tokenizer.save_pretrained(output_path)
    
    with open(output_path / 'label_encoder.json', 'w', encoding='utf-8') as f:
        json.dump({'classes': label_encoder.classes_.tolist()}, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 모델 저장 완료: {output_path}")

if __name__ == "__main__":
    main()
