# inference.py
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 공통 모듈 임포트
from common import CONFIG, emotion_score_map, KRMediumWithMLP, EmotionDataset, set_korean_font

def load_model_and_predict(movie_path, model_dir):
    print(f"\n🎬 영화 감정 분석: {movie_path}")
    model_dir = Path(model_dir)
    
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
        raise ValueError("영화 파일에 'text' 필드가 없습니다.")
    
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
    return texts, pred_emotions

def analyze_emotion_shifts(predictions, texts, window_size=10, threshold=0.8):
    print(f"\n🔄 감정 전환점 분석 시작")
    scores = np.array([emotion_score_map.get(e, 0) for e in predictions])
    n = len(scores)
    shift_points = []
    smoothed_scores = pd.Series(scores).rolling(window=window_size, min_periods=1).mean().values
    
    for i in range(window_size, n - window_size):
        prev_avg = np.mean(smoothed_scores[i - window_size:i])
        next_avg = np.mean(smoothed_scores[i:i + window_size])
        change_magnitude = abs(next_avg - prev_avg)
        
        if (prev_avg * next_avg < 0 and change_magnitude >= threshold):
            shift_type = "긍정 ➡️ 부정" if prev_avg > next_avg else "부정 ➡️ 긍정"
            is_new = all(abs(p['index'] - i) >= window_size // 2 for p in shift_points)
            
            if is_new:
                shift_points.append({
                    'index': i, 'type': shift_type,
                    'prev': f"{prev_avg:.2f}", 'next': f"{next_avg:.2f}",
                    'context': texts[i]
                })

    if shift_points:
        print(f"✅ 총 {len(shift_points)}개 전환점 발견")
        for p in shift_points:
            print(f"   * 장면 {p['index']} ({p['type']}): {p['context'][:30]}...")
    return shift_points

def plot_emotion_flow(predictions, title, mode='segment'):
    set_korean_font()
    scores = [emotion_score_map.get(e, 0) for e in predictions]
    
    if mode == 'smooth':
        window_size = max(10, len(scores) // 20)
        smoothed = pd.Series(scores).rolling(window=window_size, center=True).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(smoothed, color='steelblue', label='감정 흐름')
        plt.fill_between(range(len(smoothed)), smoothed, alpha=0.3, color='steelblue')
        plt.title(f"'{title}' 감정 흐름 (이동평균)")
        plt.legend()
    else:
        # 구간별 분석 로직 (간소화)
        n = len(predictions)
        indices = [
            (0, 1),                             # 1. 처음 (딱 첫 번째 문장)
            (0, int(n * 0.25)),                 # 2. 기 (0~25%)
            (int(n * 0.25), int(n * 0.5)),      # 3. 승 (25~50%)
            (int(n * 0.5), int(n * 0.75)),      # 4. 전 (50~75%)
            (int(n * 0.75), n),                 # 5. 결 (75~100%)
            (n - 1, n)]                         # 6. 후반 (딱 마지막 문장)
        labels = ['Start', '기', '승', '전', '결', 'End']
        
        avgs, doms = [], []
        for start, end in indices:
            part = predictions[start:end]
            if not part: 
                avgs.append(0); doms.append("없음")
                continue
            
           # --- [여기부터 수정되었습니다] ---
            
            # 1. 감정별 파워(강도) 누적 계산
            emotion_power = {}
            
            for emo in part:
                # 감정의 점수(절댓값)를 가져옴 (예: 슬픔 -0.8 -> 0.8점 / 놀람 -0.1 -> 0.1점)
                score = abs(emotion_score_map.get(emo, 0))
                
                # 중립(0점)은 무시하거나 영향력 없음
                if score == 0: continue
                
                # 단순히 개수(+1)를 세는 게 아니라, 점수(+score)를 더함!
                emotion_power[emo] = emotion_power.get(emo, 0) + score
            
            # 2. 누적 점수가 가장 높은 감정을 '지배적 감정'으로 선정
            if emotion_power:
                # 딕셔너리에서 value(점수 총합)가 가장 큰 key(감정 이름)를 찾음
                dominant_emotion = max(emotion_power, key=emotion_power.get)
                doms.append(dominant_emotion)
            else:
                # 점수 있는 감정이 하나도 없으면 중립
                doms.append("중립")
                
            # 3. 그래프에 찍을 평균 점수 계산 (이건 기존 유지)
            # (여전히 가중 평균을 쓰고 싶다면 아래처럼 유지, 단순 평균이면 mean으로 변경 가능)
            avg_score = np.mean([emotion_score_map.get(e, 0) for e in part])
            avgs.append(avg_score)

            # --- [여기까지 수정] ---

        plt.figure(figsize=(12, 6))
        plt.plot(labels, avgs, marker='o', color='gray')
        for x, y, d in zip(labels, avgs, doms):
            plt.text(x, y + 0.1, f"{d}\n({y:.2f})", ha='center')
        plt.title(f"'{title}' 구간별 감정 흐름")
        plt.ylim(-1.2, 1.2)

    plt.axhline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{title}_{mode}.png")
    plt.show()

def main():
    model_path = Path(CONFIG['output_dir']) / 'final_model_mlp'
    movie_path = CONFIG['analyze_movie_path']
    
    if model_path.exists() and Path(movie_path).exists():
        texts, preds = load_model_and_predict(movie_path, model_path)
        title = Path(movie_path).stem
        
        plot_emotion_flow(preds, title, mode='segment')
        plot_emotion_flow(preds, title, mode='smooth')
        analyze_emotion_shifts(preds, texts, CONFIG['shift_window_size'], CONFIG['shift_threshold'])
    else:
        print("⚠️ 모델 파일이나 분석할 영화 파일이 없습니다.")

if __name__ == "__main__":
    main()