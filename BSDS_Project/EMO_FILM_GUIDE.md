# Emo-Film BSDS Analysis: Step-by-Step Guide

**목적**: Emo-Film fMRI 데이터에 BSDS (Bayesian Switching Dynamical Systems) 적용
**데이터**: Emo-FilM Dataset (영화 시청 중 fMRI)
**방법론**: Taghia & Cai (2018) Nature Communications

---

## Quick Start (현실적 접근법)

현재 Python 포팅이 불완전하므로, 두 가지 경로를 제시합니다:

### Option A: MATLAB 사용 (권장 - 신뢰성 높음)
### Option B: Python 사용 (제한적 기능)

---

## Step 1: 환경 설정

### 1.1 Python 환경
```bash
# 가상환경 생성
conda create -n bsds_env python=3.9
conda activate bsds_env

# 필수 패키지
pip install numpy scipy scikit-learn nilearn nibabel matplotlib pandas
```

### 1.2 MATLAB 환경 (Option A)
```matlab
% Parallel Computing Toolbox 권장
% Statistics and Machine Learning Toolbox 필요
```

---

## Step 2: Emo-Film 데이터 전처리

### 2.1 데이터 경로 확인
```
/storage/bigdata/Emo-FilM/brain_data/derivatives/preprocessing/
├── sub-S01/ses-1/func/
│   ├── sub-S01_ses-1_task-BigBuckBunny_space-MNI_desc-ppres_bold.nii.gz
│   ├── sub-S01_ses-1_task-FirstBite_space-MNI_desc-ppres_bold.nii.gz
│   ├── sub-S01_ses-1_task-YouAgain_space-MNI_desc-ppres_bold.nii.gz
│   └── sub-S01_ses-1_task-Rest_space-MNI_desc-ppres_bold.nii.gz
├── sub-S02/...
└── ...
```

### 2.2 ROI 시계열 추출 (수정된 스크립트)

`run_extraction_emofilm.py`:
```python
#!/usr/bin/env python3
"""
Emo-Film ROI Extraction Script
영화별, 피험자별 시계열 데이터 추출
"""

import os
import sys
import numpy as np
from datetime import datetime
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

# ====== 설정 ======
BASE_DIR = '/storage/bigdata/Emo-FilM/brain_data/derivatives/preprocessing'
RESULT_DIR = './results/emofilm'
TASKS = ['BigBuckBunny', 'FirstBite', 'YouAgain', 'Rest']  # 분석할 영화

# 피험자 목록 (자동 스캔)
def get_subjects():
    subs = [d for d in os.listdir(BASE_DIR) if d.startswith('sub-')]
    return sorted(subs)

def extract_timeseries(subject, task, masker, output_dir):
    """단일 피험자, 단일 태스크 시계열 추출"""
    file_name = f"{subject}_ses-1_task-{task}_space-MNI_desc-ppres_bold.nii.gz"
    func_path = os.path.join(BASE_DIR, subject, 'ses-1', 'func', file_name)

    if not os.path.exists(func_path):
        print(f"  ⏭️ Skip: {func_path}")
        return None

    try:
        ts = masker.fit_transform(func_path)  # (Time x ROI)
        save_path = os.path.join(output_dir, f"{subject}_{task}_timeseries.npy")
        np.save(save_path, ts)
        print(f"  ✅ {subject}/{task}: {ts.shape}")
        return ts
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    # 결과 폴더
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULT_DIR, f"{timestamp}_extraction")
    os.makedirs(output_dir, exist_ok=True)

    # Atlas 설정 (Schaefer 400 parcels)
    print("🧠 Loading Schaefer Atlas...")
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    masker = NiftiLabelsMasker(
        labels_img=schaefer.maps,
        standardize=True,
        memory='nilearn_cache'
    )

    # 피험자 목록
    subjects = get_subjects()
    print(f"📋 Found {len(subjects)} subjects")

    # 추출 루프
    for task in TASKS:
        print(f"\n🎬 Processing Task: {task}")
        for sub in subjects:
            extract_timeseries(sub, task, masker, output_dir)

    print(f"\n🎉 Done! Results in: {output_dir}")

if __name__ == '__main__':
    main()
```

실행:
```bash
python run_extraction_emofilm.py
```

---

## Step 3: BSDS 분석

### Option A: MATLAB (권장)

#### 3A.1 데이터 준비
```matlab
% data_prep.m
clear all; close all;

% 피험자 목록
subjects = {'sub-S01', 'sub-S02', 'sub-S03'};  % 실제 목록으로 교체
task = 'BigBuckBunny';

% 데이터 로드
data = {};
for i = 1:length(subjects)
    ts = readNPY(sprintf('results/emofilm/%s_%s_timeseries.npy', subjects{i}, task));
    data{i} = ts';  % (ROI x Time) 형태로 전치
    fprintf('Loaded %s: %d x %d\n', subjects{i}, size(data{i}));
end
```

#### 3A.2 BSDS 실행
```matlab
% run_bsds.m
addpath('Taghia_Cai_NatureComm_2018-main');
addpath('Taghia_Cai_NatureComm_2018-main/functions');

% 설정
max_nstates = 8;   % 최대 상태 수
max_ldim = 20;     % 잠재 차원

% 옵션
opt.n_iter = 100;
opt.n_init_iter = 10;
opt.tol = 1e-3;
opt.noise = 0;
opt.n_init_learning = 5;

% 실행!
fprintf('Starting BSDS fitting...\n');
model = BayesianSwitchingDynamicalSystems(data, max_nstates, max_ldim, opt);

% 결과 저장
save('bsds_result_emofilm.mat', 'model');
fprintf('Done! Saved to bsds_result_emofilm.mat\n');
```

#### 3A.3 결과 분석
```matlab
% analyze_results.m
load('bsds_result_emofilm.mat');

% 주요 결과
fprintf('=== BSDS Results ===\n');
fprintf('Dominant states: %s\n', mat2str(model.id_of_dominant_states_group_wise));
fprintf('Fractional occupancy:\n');
disp(model.fractional_occupancy_group_wise);
fprintf('Mean lifetime:\n');
disp(model.mean_lifetime_group_wise);

% 시각화: 상태 전이 행렬
figure;
imagesc(model.state_transition_probabilities);
colorbar;
title('State Transition Matrix');
xlabel('To State'); ylabel('From State');

% 시각화: 상태 시계열 (피험자 1)
figure;
plot(model.temporal_evolution_of_states{1});
title('State Sequence - Subject 1');
xlabel('Time (TR)'); ylabel('State');
```

### Option B: Python (제한적)

현재 Python 구현은 불완전하지만, 기본적인 HMM 수준의 분석은 가능합니다:

```python
# run_bsds_python.py
import numpy as np
import sys
sys.path.append('bsds')
from bsds_model import BSDSModel

# 데이터 로드
subjects = ['sub-S01', 'sub-S02', 'sub-S03']
task = 'BigBuckBunny'

data_list = []
for sub in subjects:
    ts = np.load(f'results/emofilm/{sub}_{task}_timeseries.npy')
    data_list.append(ts.T)  # (ROI x Time)

# 모델 피팅
model = BSDSModel(n_states=5, max_ldim=10, n_iter=50)
model.fit(data_list)

# 결과 출력
print("Transition Matrix:")
print(model.stran)
print("\nLog Likelihood History:")
print(model.Fhist)

# 주의: Python 버전은 AR 동역학이 없으므로 결과 해석에 주의
```

---

## Step 4: 결과 해석

### 4.1 핵심 출력 변수

| Variable | Shape | Description |
|----------|-------|-------------|
| `temporal_evolution_of_states` | {S}(1 x T) | 각 피험자의 시간별 상태 |
| `state_transition_probabilities` | (K x K) | 상태 전이 확률 행렬 |
| `fractional_occupancy` | (1 x K) | 각 상태의 점유율 |
| `mean_lifetime` | (1 x K) | 각 상태의 평균 체류 시간 |
| `estimated_covariance` | {K}(D x D) | 각 상태의 공분산 행렬 |

### 4.2 해석 예시

```matlab
% 영화 장면과 상태 매핑
TR = 2;  % TR in seconds
states = model.temporal_evolution_of_states{1};
time_sec = (1:length(states)) * TR;

% 특정 시간대 (예: 영화 시작 후 2분)
t_interest = 120;  % seconds
state_at_t = states(round(t_interest/TR));
fprintf('At t=%ds: Brain state = %d\n', t_interest, state_at_t);
```

### 4.3 그룹 비교 (예: 불안 높음 vs 낮음)

```matlab
% 그룹별 점유율 비교
occ_high_anxiety = model_high.fractional_occupancy_group_wise;
occ_low_anxiety = model_low.fractional_occupancy_group_wise;

% 통계 검정
[h, p] = ttest2(occ_high_anxiety, occ_low_anxiety);
fprintf('State occupancy difference p-value: %.4f\n', p);
```

---

## Step 5: 초록 작성을 위한 핵심 결과

### 5.1 보고할 주요 지표
- Number of dominant states: `model.id_of_dominant_states_group_wise`
- State occupancy: `model.fractional_occupancy_group_wise`
- Mean dwell time: `model.mean_lifetime_group_wise`
- Transition patterns: `model.state_transition_probabilities`

### 5.2 초록 템플릿

```
[배경] 영화 시청 중 뇌 활동은 동적으로 변화하며, 이러한 동적 상태는
정서 처리와 관련될 수 있다.

[방법] Emo-FilM 데이터셋(N=XX명)에 Bayesian Switching Dynamical
Systems (BSDS; Taghia & Cai, 2018)를 적용하여 영화 시청 중 뇌
상태 전환 패턴을 분석하였다. Schaefer 400 ROI를 사용하여
시계열을 추출하고, K=[X]개 상태로 모델링하였다.

[결과] [X]개의 주요 뇌 상태가 확인되었으며, 각 상태의 평균
체류 시간은 [X-X]초였다. 특정 영화 장면에서 상태 전환이
집중되는 패턴을 발견하였다.

[결론] BSDS 분석을 통해 영화 시청 중 뇌 활동의 동적 특성을
정량화할 수 있으며, 이는 정서 처리 연구에 새로운 관점을 제공한다.
```

---

## Troubleshooting

### Q: MATLAB에서 메모리 에러
```matlab
% 메모리 절약 옵션
opt.n_init_learning = 3;  % 줄이기
max_ldim = 10;  % 줄이기
```

### Q: Python에서 수렴하지 않음
```python
# 더 많은 iteration
model = BSDSModel(n_states=5, max_ldim=10, n_iter=200)
```

### Q: ROI 추출 시 에러
```bash
# Nilearn 캐시 초기화
rm -rf nilearn_cache/
```

---

## 참고 문헌

1. Taghia, J., & Cai, M. B. et al. (2018). Uncovering hidden brain state dynamics that regulate performance and decision-making during cognition. Nature Communications.

2. Schaefer, A. et al. (2018). Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI.

---

*Last updated: 2025-12-15*
