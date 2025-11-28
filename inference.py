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

def find_movie_file_by_title(target_title, search_dir):
    """
    지정된 디렉토리(search_dir)의 json 파일들을 순회하며
    'title' 값이 target_title과 일치하는 파일 경로를 반환합니다.
    """
    search_path = Path(search_dir)
    if not search_path.exists():
        print(f"⚠️ 경고: 검색할 폴더가 존재하지 않습니다: {search_path}")
        return None

    print(f"🔍 '{target_title}' 영화 파일을 찾는 중... (폴더: {search_path})")
    
    # 폴더 내의 모든 .json 파일 검색
    for file_path in search_path.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # JSON 내 title 키 확인 (공백 제거 후 비교)
                if 'title' in data and data['title'].strip() == target_title.strip():
                    return file_path
        except Exception as e:
            # 파일 읽기 에러 시 건너뜀
            continue
            
    return None

def load_model_and_predict(movie_path, model_dir):
    print(f"\n🎬 영화 감정 분석 시작: {movie_path}")
    model_dir = Path(model_dir)
    
    # 1. 라벨 정보 로드
    with open(model_dir / 'label_encoder.json', 'r', encoding='utf-8') as f:
        label_data = json.load(f)
        id2label = {i: label for i, label in enumerate(label_data['classes'])}
        num_labels = len(label_data['classes'])
        
    # 2. 모델 로드
    model = KRMediumWithMLP(CONFIG['model_name'], num_labels)
    try:
        model.load_state_dict(torch.load(model_dir / 'pytorch_model.bin'), strict=False)
    except Exception as e:
        print(f"⚠️ 모델 로드 중 경고(무시 가능): {e}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 3. 데이터 로드
    with open(movie_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    # 데이터 구조에 따른 텍스트 추출 로직
    if 'units' in data:
        for unit in data.get('units', []):
            for script in unit.get('story_scripts', []):
                content = script.get('content')
                if content:
                    texts.append(content)
    elif 'text' in data:
        texts = data['text']
    
    if not texts:
        raise ValueError(f"영화 파일에서 텍스트를 찾을 수 없습니다. 구조를 확인해주세요.")
    
    print(f"📝 총 {len(texts)}개 장면(문장) 분석 중...")
    
    # 4. 추론 실행
    dummy_labels = [0] * len(texts)
    pred_dataset = EmotionDataset(texts, dummy_labels, tokenizer, CONFIG['max_length'])
    pred_dataloader = DataLoader(pred_dataset, batch_size=CONFIG['analysis_batch_size'], shuffle=False)
    
    all_pred_ids = []
    
    with torch.no_grad():
        for batch in tqdm(pred_dataloader, desc="감정을 읽는 중"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)[0]
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            all_pred_ids.extend(pred_ids)

    pred_emotions = [id2label[i] for i in all_pred_ids]
    return texts, pred_emotions

def analyze_emotion_shifts(predictions, texts, window_size=10, threshold=0.8):
    print(f"\n🔄 감정 전환점 분석 시작 (구간: {window_size}, 임계값: {threshold})")
    scores = np.array([emotion_score_map.get(e, 0) for e in predictions])
    n = len(scores)
    shift_points = []
    
    # 이동 평균 계산
    smoothed_scores = pd.Series(scores).rolling(window=window_size, min_periods=1).mean().values
    
    for i in range(window_size, n - window_size):
        prev_avg = np.mean(smoothed_scores[i - window_size:i])
        next_avg = np.mean(smoothed_scores[i:i + window_size])
        
        # 감정의 부호가 바뀌고 변화량이 큰 경우
        change_magnitude = abs(next_avg - prev_avg)
        
        if (prev_avg * next_avg < 0 and change_magnitude >= threshold):
            shift_type = "긍정(행복) ➡️ 부정(위기)" if prev_avg > next_avg else "부정(위기) ➡️ 긍정(해소)"
            
            # 중복 방지
            is_new = all(abs(p['index'] - i) >= window_size // 2 for p in shift_points)
            
            if is_new:
                shift_points.append({
                    'index': i, 
                    'type': shift_type,
                    'prev_score': float(prev_avg),
                    'next_score': float(next_avg),
                    'context': texts[i]
                })

    if shift_points:
        print(f"✅ 총 {len(shift_points)}개의 주요 감정 전환점 발견!")
        for p in shift_points:
            print(f"  🚩 장면 #{p['index']} [{p['type']}]")
            print(f"     대사: \"{p['context'][:40]}...\"")
            print(f"     변화: {p['prev_score']:.2f} -> {p['next_score']:.2f}")
    else:
        print("특별한 감정 반전 포인트가 발견되지 않았습니다.")
        
    return shift_points

def plot_emotion_flow(predictions, title, output_dir):
    set_korean_font()
    scores = [emotion_score_map.get(e, 0) for e in predictions]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    ax1 = axes[0]
    window_size = max(10, len(scores) // 20)
    smoothed = pd.Series(scores).rolling(window=window_size, center=True).mean()
    
    # 배경 색상 (긍정/부정 영역)
    ax1.axhspan(0, 1.1, facecolor='green', alpha=0.05)
    ax1.axhspan(-1.1, 0, facecolor='red', alpha=0.05)
    
    ax1.plot(smoothed, color='steelblue', linewidth=2, label='감정 흐름')
    ax1.fill_between(range(len(smoothed)), smoothed, alpha=0.3, color='steelblue')
    
    ax1.set_title(f"'{title}' 감정 흐름 (Time Series)", fontsize=15, fontweight='bold')
    ax1.set_xlabel("장면 진행 (Time)", fontsize=12)
    ax1.set_ylabel("감정 점수", fontsize=12)
    ax1.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax1.set_yticklabels(['공포/분노', '슬픔', '중립', '약한 긍정', '기쁨'])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left')
    ax1.set_ylim(-1.2, 1.2)

    ax2 = axes[1]
    n = len(predictions)
    sections = {
        '도입부': (0, int(n*0.2)),
        '전개': (int(n*0.2), int(n*0.5)),
        '위기(절정)': (int(n*0.5), int(n*0.8)),
        '결말': (int(n*0.8), n)
    }
    
    labels = []
    avgs = []
    doms = []
    
    for name, (start, end) in sections.items():
        part = predictions[start:end]
        if not part: 
            avg_score = 0
            dominant = "없음"
        else:
            avg_score = np.mean([emotion_score_map.get(e, 0) for e in part])
            emotion_power = {}
            for emo in part:
                score = abs(emotion_score_map.get(emo, 0))
                if score == 0: score = 0.1
                emotion_power[emo] = emotion_power.get(emo, 0) + score
            dominant = max(emotion_power, key=emotion_power.get) if emotion_power else "없음"
            
        labels.append(name)
        avgs.append(avg_score)
        doms.append(dominant)

    # 꺾은선 (흐름 파악용)
    ax2.plot(labels, avgs, marker='o', color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # 막대 그래프
    colors = ['red' if s < 0 else 'green' for s in avgs]
    bars = ax2.bar(labels, avgs, color=colors, alpha=0.6, width=0.5)
    
    # 텍스트 라벨
    for rect, dom, val in zip(bars, doms, avgs):
        height = rect.get_height()
        pos = height + 0.05 if height >= 0 else height - 0.15
        ax2.text(rect.get_x() + rect.get_width()/2.0, pos, 
                 f"{dom}\n({val:.2f})", ha='center', va='center', fontweight='bold', fontsize=11)

    ax2.set_title(f"'{title}' 기승전결 구간 분석", fontsize=15, fontweight='bold')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylim(-1.2, 1.2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    
    # 파일로도 저장
    save_path = output_dir / f"{title}_combined_analysis.png"
    plt.savefig(save_path)
    print(f"📊 그래프 이미지 저장됨: {save_path}")
    
    plt.show()

def save_results_to_json(texts, preds, title, output_dir):
    output_data = []
    for t, p in zip(texts, preds):
        output_data.append({
            'content': t,
            'emotion': p,
            'score': emotion_score_map.get(p, 0)
        })
    
    save_path = Path(output_dir) / f"{title}_analyzed_result.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"💾 상세 분석 결과 저장됨: {save_path}")

def main():
    model_path = Path(CONFIG['output_dir']) / 'final_model_mlp'
    output_dir = Path(CONFIG['output_dir'])
    
    test_data_dir = "./test" 
    
    if not model_path.exists():
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        return

    # 1. 사용자 입력 받기
    while True:
        target_title = input("\n📽️ 분석할 영화 제목을 입력하세요 (종료하려면 q 입력): ").strip()
        
        if target_title.lower() == 'q':
            print("프로그램을 종료합니다.")
            break
            
        if not target_title:
            print("제목을 입력해주세요.")
            continue

        # 2. 파일 찾기
        found_file_path = find_movie_file_by_title(target_title, test_data_dir)
        
        if found_file_path:
            print(f"✅ 파일을 찾았습니다: {found_file_path}")
            
            # 3. 추론 및 분석 실행
            texts, preds = load_model_and_predict(found_file_path, model_path)
            
            # 4. 결과 저장
            save_results_to_json(texts, preds, target_title, output_dir)
            
            # 5. 시각화 (수정됨: 한 번 호출로 통합 그래프 생성)
            plot_emotion_flow(preds, target_title, output_dir)
            
            # 6. 텍스트 전환점 분석
            analyze_emotion_shifts(preds, texts, CONFIG['shift_window_size'], CONFIG['shift_threshold'])
            
            print(f"\n🎉 '{target_title}' 분석이 완료되었습니다!")
        else:
            print(f"❌ '{test_data_dir}' 폴더 내에서 제목이 '{target_title}'인 영화를 찾을 수 없습니다.")
            print("   (띄어쓰기가 정확한지, 파일이 해당 폴더에 있는지 확인해주세요)")

if __name__ == "__main__":
    main()

