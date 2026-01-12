# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import pickle

# ========================================================
# [핵심 수정] bsds 폴더 안으로 직접 진입하게 경로 설정
# ========================================================
# 현재 파일(run_analysis.py)이 있는 위치에서 'bsds' 폴더를 찾습니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
bsds_package_path = os.path.join(current_dir, 'bsds')

# 'bsds' 폴더 자체를 sys.path에 추가합니다.
sys.path.append(bsds_package_path)
print(f"📂 패키지 경로 강제 추가: {bsds_package_path}")

# [핵심 수정] from bsds.bsds_model -> from bsds_model
# 이제 bsds 폴더 안에 들어와 있으므로, 'bsds.'을 떼고 바로 파일명을 부릅니다.
try:
    from bsds_model import BSDSModel 
    print("✅ Custom BSDSModel 로딩 성공! (bsds_utils도 찾을 수 있음)")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit()

# ========================================================
# 1. 설정 및 데이터 찾기
# ========================================================
SUBJECT_ID = 'sub-S01' 
# run_analysis.py가 root에 있으므로 results는 code/results 또는 ./results 확인 필요
# tree 구조상 results는 'code' 폴더 안에 있습니다.
RESULTS_ROOT = os.path.join(current_dir, 'code', 'results')

# 가장 최근 결과 폴더 찾기
search_pattern = os.path.join(RESULTS_ROOT, '*_Schaefer400_Results')
all_dirs = glob.glob(search_pattern)

if not all_dirs:
    print(f"❌ 결과 폴더가 없습니다. 경로 확인: {RESULTS_ROOT}")
    print("   (run_extraction.py를 먼저 실행했는지 확인해주세요)")
    sys.exit()

latest_dir = max(all_dirs, key=os.path.getmtime)
data_path = os.path.join(latest_dir, f"{SUBJECT_ID}_timeseries.npy")

print(f"📂 타겟 데이터 폴더: {latest_dir}")
if not os.path.exists(data_path):
    print(f"❌ 데이터 파일 없음: {data_path}")
    sys.exit()

# ========================================================
# 2. 데이터 로드 및 피팅
# ========================================================
timeseries = np.load(data_path)
print(f"📊 원본 데이터: {timeseries.shape} (Time x ROI)")

# Transpose (ROI x Time)
data_for_model = timeseries.T
print(f"🔄 모델 입력용: {data_for_model.shape} (ROI x Time)")

# 모델 설정
n_states = 5
max_ldim = 5  # 필수 파라미터
n_iter = 50

print(f"\n🧠 BSDS 피팅 시작 (K={n_states}, Ldim={max_ldim})...")

# 모델 생성
model = BSDSModel(n_states=n_states, max_ldim=max_ldim, n_iter=n_iter)

# 학습 (리스트로 묶어서 전달!)
try:
    model.fit([data_for_model])
    print("🎉 학습 완료!")
except Exception as e:
    print(f"💥 학습 중 에러 발생: {e}")
    sys.exit()

# ========================================================
# 3. 결과 저장
# ========================================================
save_name = f"{SUBJECT_ID}_BSDS_k{n_states}_ldim{max_ldim}_result.pkl"
save_path = os.path.join(latest_dir, save_name)

result_data = {
    'subject': SUBJECT_ID,
    'Wa': model.Wa,
    'Wpi': model.Wpi,
    'Fhist': model.Fhist,
    'model_params': model.__dict__ 
}

with open(save_path, 'wb') as f:
    pickle.dump(result_data, f)

print(f"💾 결과 저장 완료: {save_path}")