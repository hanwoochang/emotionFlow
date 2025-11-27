# train.py
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback

# 공통 모듈 임포트
from common import CONFIG, EMOTION_MAPPING, KRMediumWithMLP, EmotionDataset, set_korean_font

# 데이터 로드 함수 (학습용)
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

def plot_training_history(history, output_dir):
    set_korean_font()
    print("\n📊 학습 결과 시각화 중...")
    
    train_loss = [x['loss'] for x in history if 'loss' in x]
    train_steps = [x['step'] for x in history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    eval_steps = [x['step'] for x in history if 'eval_loss' in x]
    eval_acc = [x['eval_accuracy'] for x in history if 'eval_accuracy' in x]

    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_loss, label='Train Loss', alpha=0.6, color='salmon')
    plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='o', color='steelblue')
    plt.title('Loss 변화')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if eval_acc: plt.plot(eval_steps, eval_acc, label='Accuracy', marker='s')
    plt.title('성능 지표 변화')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'training_history.png')
    print(f"💾 학습 그래프 저장 완료")
    plt.show()

def main():
    print("\n🚀 감정 분류 모델 학습 시작\n")
    
    X_train, y_train_text = load_emotion_data('./train_data/label/', CONFIG['excluded_emotions'])
    X_val, y_val_text = load_emotion_data('./validation_data/label/', CONFIG['excluded_emotions'])

    valid_emotions = set(y_train_text)
    X_val_filtered = [x for x, y in zip(X_val, y_val_text) if y in valid_emotions]
    y_val_filtered = [y for y in y_val_text if y in valid_emotions]
    
    if not X_val_filtered:
        X_val_filtered, y_val_filtered = X_train[:100], y_train_text[:100]
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_val = label_encoder.transform(y_val_filtered)
    num_labels = len(label_encoder.classes_)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = KRMediumWithMLP(model_name=CONFIG['model_name'], num_labels=num_labels)

    train_dataset = EmotionDataset(X_train, y_train, tokenizer, CONFIG['max_length'])
    val_dataset = EmotionDataset(X_val_filtered, y_val, tokenizer, CONFIG['max_length'])

    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
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
    
    trainer.train()
    plot_training_history(trainer.state.log_history, CONFIG['output_dir'])

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