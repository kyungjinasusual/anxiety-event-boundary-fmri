# 데이터 로딩 및 ROI 추출 스크립트
# 작성일: 2025-12-11

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import load_img

# ==========================================
# 1. 설정 (Settings)
# ==========================================
BASE_DIR = '/storage/bigdata/Emo-FilM/brain_data/derivatives/preprocessing'
RESULT_DIR = './results'  # 결과 저장할 곳
# SUB_LIST = ['sub-S01']    # 테스트할 피험자 리스트 (나중엔 자동으로 긁어오게 수정)

# [수정] 커맨드라인에서 subject 입력받기
# 사용법: python run_extraction.py sub-S01
if len(sys.argv) > 1:
    subject_id = sys.argv[1] # 터미널에서 던져준 첫 번째 단어를 받음
    SUB_LIST = [subject_id]
else:
    print("피험자 ID를 입력해주세요! (예: python run_extraction.py sub-S01)")
    sys.exit()

# 타임스탬프 생성 (폴더명용)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(RESULT_DIR, f"{timestamp}_Schaefer400_Results")
os.makedirs(output_dir, exist_ok=True)
print(f"📂 결과 폴더 생성됨: {output_dir}")

# ==========================================
# 2. 아틀라스 불러오기 (Schaefer 400 parcel, 7 networks)
# ==========================================
print("🧠 Schaefer Atlas 로딩 중...")
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
# resolution_mm은 데이터 해상도(2mm or 3mm)에 맞춰야 함. 보통 2mm 많이 씀.

# Masker 객체 생성 (이 녀석이 추출, 스무딩, 잡음제거 다 해줌)
masker = NiftiLabelsMasker(
    labels_img=schaefer.maps,
    standardize=True,      # 신호 정규화 (z-score)
    memory='nilearn_cache',# 캐시 사용 (속도 향상)
    verbose=5              # 진행상황 수다스럽게 출력
)

# ==========================================
# 3. 데이터 로딩 및 추출 루프
# ==========================================
# 분석할 영화 제목을 여기서 선택하세요!
# 옵션: 'BigBuckBunny', 'FirstBite', 'YouAgain', 'Rest'
TARGET_TASK = 'BigBuckBunny' 

for sub in SUB_LIST:
    print(f"\n🚀 Processing {sub} - Task: {TARGET_TASK}...")
    
    # (1) 파일 경로 찾기 (스크린샷 기반 수정)
    # 실제 파일명 패턴: sub-S01_ses-1_task-BigBuckBunny_space-MNI_desc-ppres_bold.nii.gz
    file_name = f"{sub}_ses-1_task-{TARGET_TASK}_space-MNI_desc-ppres_bold.nii.gz"
    func_path = os.path.join(BASE_DIR, sub, 'ses-1', 'func', file_name)
    
    # 주의: 스크린샷에는 표준 confounds 파일(.tsv)이 보이지 않습니다. 
    # 만약 잡음 제거용 파일이 없다면 None으로 설정해야 에러가 안 납니다.
    # (보통 같은 폴더에 있어야 하는데, 없으면 motion_bold.tsv.gz 등을 써야 할 수도 있습니다)
    confounds_path = None 
    
    # 만약 confounds 파일이 있다면 아래 주석을 풀고 경로를 맞춰주세요.
    # confounds_name = f"{sub}_ses-1_task-{TARGET_TASK}_desc-confounds_timeseries.tsv"
    # confounds_path = os.path.join(BASE_DIR, sub, 'ses-1', 'func', confounds_name)

    if not os.path.exists(func_path):
        print(f"❌ 파일 없음: {func_path}")
        continue

    # (2) ROI 시계열 추출
    try:
        # confounds가 None이면 잡음 제거 없이 신호만 추출합니다.
        time_series = masker.fit_transform(func_path, confounds=confounds_path)
        print(f"✅ 추출 완료! 데이터 크기: {time_series.shape} (시간 x ROI수)")
        
        # (3) 상관관계 행렬 계산 (Connectivity Matrix)
        correlation_matrix = np.corrcoef(time_series.T) # 전치시켜서 (ROI x ROI) 구함
        
        # (4) 저장
        save_name_ts = f"{sub}_timeseries.npy"
        save_name_corr = f"{sub}_correlation_matrix.npy"
        
        np.save(os.path.join(output_dir, save_name_ts), time_series)
        np.save(os.path.join(output_dir, save_name_corr), correlation_matrix)
        print(f"💾 저장 완료: {sub}")
        
    except Exception as e:
        print(f"💥 에러 발생 ({sub}): {e}")

print("\n🎉 모든 작업 종료!")