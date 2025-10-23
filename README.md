# Event Boundary Detection in Trait/State Anxiety using fMRI Transformer

## 연구 개요

**목표**: Resting-state fMRI 데이터에서 SwiFT (fMRI Transformer)를 활용하여 특성/상태 불안과 사건 경계(event boundary)의 관계를 규명

**데이터**: Open datasets 활용 (참가자 직접 모집 없음)

**핵심 질문**:
1. 불안 수준이 사건 경계 탐지에 어떤 영향을 미치는가?
2. Transformer 기반 표현 학습이 전통적 방법론 대비 어떤 이점이 있는가?
3. SwiFT의 attention mechanism을 통해 뇌 영역 간 상호작용을 어떻게 해석할 수 있는가?

## 방법론

### 1. SwiFT (fMRI Transformer)
- **장점**: 장거리 시공간 의존성 포착, 동적 뇌 상태 모델링
- **용도**: Event boundary 자동 탐지, resting-state representation 학습

### 2. Hybrid Interpretation Approach
```
SwiFT (Pattern Detection)
    ↓
Event Boundary Detection
    ↓
Traditional Analysis (Interpretation)
    ├─ Seed-based connectivity
    ├─ Network analysis (DMN, SN, CEN)
    └─ ROI-based activation

Validation
    ├─ Attention weights visualization
    ├─ Perturbation analysis
    └─ Behavioral correlation (불안 척도)
```

### 3. Baseline Methods
- Hidden Markov Model (HMM)
- Sliding window correlation
- ICA-based segmentation

## 프로젝트 구조

```
anxiety-event-boundary-fmri/
├── data/                   # fMRI 데이터
│   ├── raw/               # 원본 데이터
│   ├── processed/         # 전처리된 데이터
│   └── derivatives/       # 분석 결과
│
├── models/                # 모델 관련
│   ├── swift/            # SwiFT 모델
│   ├── baselines/        # 비교 모델 (HMM, etc)
│   └── checkpoints/      # 학습된 가중치
│
├── analysis/             # 분석 스크립트
│   ├── event_detection/  # 사건 경계 탐지
│   ├── connectivity/     # 연결성 분석
│   └── interpretation/   # 해석 및 시각화
│
├── results/              # 결과물
│   ├── figures/         # 그림
│   ├── tables/          # 표
│   └── statistics/      # 통계 결과
│
├── papers/              # 논문 관련
│   ├── references/      # 참고문헌
│   └── drafts/         # 초안
│
├── src/                 # 소스 코드
│   ├── preprocessing/   # 데이터 전처리
│   ├── training/        # 모델 학습
│   ├── evaluation/      # 평가
│   └── visualization/   # 시각화
│
├── configs/             # 설정 파일
└── notebooks/           # Jupyter 노트북
```

## 실험 워크플로우

### Phase 1: Data Preparation
1. fMRI 데이터 전처리 (motion correction, normalization)
2. 불안 척도 수집 (STAI-T, STAI-S)
3. 데이터셋 분할 (train/val/test)

### Phase 2: Model Training
1. SwiFT 사전학습 모델 로드
2. Fine-tuning for event boundary detection
3. Baseline 모델 학습 (HMM, sliding window)

### Phase 3: Analysis & Interpretation
1. Event boundary 탐지 성능 비교
2. Attention weights 분석
3. 전통적 connectivity 분석
4. 불안 척도와 상관관계 분석

### Phase 4: Validation
1. Cross-validation
2. 독립 데이터셋 테스트
3. Clinical relevance 평가

## 설치 및 환경 설정

### Requirements
```bash
python >= 3.8
pytorch >= 2.0
nilearn
nibabel
scipy
pandas
matplotlib
seaborn
```

### Setup
```bash
# 환경 생성
conda create -n anxiety-fmri python=3.9
conda activate anxiety-fmri

# 의존성 설치
pip install -r requirements.txt

# SwiFT 설치
git clone https://github.com/athms/swift-fmri
cd swift-fmri
pip install -e .
```

## 사용법

### 1. 데이터 전처리
```bash
python src/preprocessing/preprocess_fmri.py \
    --input data/raw \
    --output data/processed
```

### 2. 모델 학습
```bash
python src/training/train_swift.py \
    --config configs/swift_config.yaml \
    --data data/processed \
    --output models/swift
```

### 3. Event Boundary 탐지
```bash
python analysis/event_detection/detect_boundaries.py \
    --model models/swift/best_model.pt \
    --data data/processed \
    --output results
```

### 4. 해석 및 분석
```bash
python analysis/interpretation/analyze_attention.py \
    --model models/swift/best_model.pt \
    --output results/figures
```

## 예상 결과

1. **SwiFT vs Baselines**: Event boundary 탐지 정확도 비교
2. **Attention Maps**: 불안 상태에서의 뇌 영역 상호작용 시각화
3. **Connectivity Analysis**: 전통적 분석과의 교차 검증
4. **Clinical Correlation**: 불안 척도와 뇌 활동 패턴의 관계

## 연구 일정

- [ ] Week 1-2: 데이터 수집 및 전처리
- [ ] Week 3-4: SwiFT fine-tuning
- [ ] Week 5-6: Baseline 모델 학습 및 비교
- [ ] Week 7-8: 해석 및 분석
- [ ] Week 9-10: 논문 작성
- [ ] Week 11-12: 리뷰 및 수정

## 참고문헌

1. SwiFT: Swin 4D fMRI Transformer (https://github.com/athms/swift-fmri)
2. Event Segmentation Theory
3. Anxiety and Brain Connectivity Literature

## 라이선스

MIT License

## 저자

Kyungjin Oh (castella@snu.ac.kr)

---

**Note**: 이 프로젝트는 멀티-에이전트 연구 시스템을 통해 관리됩니다.
