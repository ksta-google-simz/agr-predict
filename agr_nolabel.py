import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

from deepface import DeepFace
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

def compare_agr_no_label(original_dir, anonymized_dir):
    # 결과를 저장할 변수 초기화
    differences = []
    total_predictions = 0
    
    # 이미지 파일 목록 가져오기
    image_files = glob.glob(os.path.join(anonymized_dir, "*.jpg"))
    
    # 메트릭을 저장할 변수 초기화
    orig_metrics = {
        'age': [],
        'gender': [],
        'race': []
    }
    anon_metrics = {
        'age': [],
        'gender': [],
        'race': []
    }
    
    for anon_path in tqdm(image_files):
        try:
            # 파일 이름으로 원본 이미지 경로 찾기
            img_id = os.path.basename(anon_path)
            orig_path = os.path.join(original_dir, img_id)
            
            # 각 이미지에 대한 AGR 분석
            orig_pred = DeepFace.analyze(orig_path, actions=['age', 'gender', 'race'], 
                                       align=True, detector_backend='retinaface')
            anon_pred = DeepFace.analyze(anon_path, actions=['age', 'gender', 'race'], 
                                       align=True, detector_backend='retinaface')
            
            if not orig_pred or not anon_pred or \
               not isinstance(orig_pred, list) or not isinstance(anon_pred, list) or \
               len(orig_pred) == 0 or len(anon_pred) == 0:
                print(f"이미지 {img_id}에 대한 예측 형식이 잘못되었습니다")
                continue
            
            orig = orig_pred[0]
            anon = anon_pred[0]
            
            # 결과 저장
            orig_metrics['age'].append(orig.get('age', 0))
            anon_metrics['age'].append(anon.get('age', 0))
            orig_metrics['gender'].append(orig.get('dominant_gender', ''))
            anon_metrics['gender'].append(anon.get('dominant_gender', ''))
            orig_metrics['race'].append(orig.get('dominant_race', ''))
            anon_metrics['race'].append(anon.get('dominant_race', ''))
            
            total_predictions += 1
            
        except Exception as e:
            print(f"이미지 {img_id} 처리 중 오류 발생: {str(e)}")
            continue
    
    # 결과 분석
    print("\n=== AGR 분석 결과 ===")
    print(f"분석된 총 이미지 수: {total_predictions}")
    
    # 나이 차이 분석
    age_diff = np.array(anon_metrics['age']) - np.array(orig_metrics['age'])
    print("\n[나이 분석]")
    print(f"평균 절대 나이 차이: {np.mean(np.abs(age_diff)):.2f}세")
    print(f"나이가 증가한 비율: {np.mean(age_diff > 0):.2%}")
    print(f"나이가 감소한 비율: {np.mean(age_diff < 0):.2%}")
    
    # 성별 분석
    gender_match = np.array(orig_metrics['gender']) == np.array(anon_metrics['gender'])
    print("\n[성별 분석]")
    print(f"성별이 유지된 비율: {np.mean(gender_match):.2%}")
    
    # 인종 분석
    race_match = np.array(orig_metrics['race']) == np.array(anon_metrics['race'])
    print("\n[인종 분석]")
    print(f"인종이 유지된 비율: {np.mean(race_match):.2%}")
    
    # 상세 변화 통계
    if not gender_match.all():
        print("\n[성별 변화 상세]")
        gender_changes = pd.DataFrame({
            '원본': orig_metrics['gender'],
            '변환': anon_metrics['gender']
        })[~gender_match]
        print(gender_changes.value_counts().to_string())
    
    if not race_match.all():
        print("\n[인종 변화 상세]")
        race_changes = pd.DataFrame({
            '원본': orig_metrics['race'],
            '변환': anon_metrics['race']
        })[~race_match]
        print(race_changes.value_counts().to_string())

if __name__ == '__main__':
    original_dir = 'celeb/original'  # 원본 이미지 디렉토리
    anonymized_dir = 'celeb/anon_10_102'  # 익명화된 이미지 디렉토리
    
    compare_agr_no_label(original_dir, anonymized_dir)
