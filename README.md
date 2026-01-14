# UNSEAM

**UN**covering **S**tate-dependent **E**vent boundaries in **A**nxiety using neural **M**odels

불안과 신경 사건 분할(neural event segmentation)의 관계를 규명하는 연구 프로젝트

## 연구 개요

**목표**: fMRI 데이터에서 뇌 상태 전환(brain state transitions)을 탐지하고, 특성/상태 불안이 사건 경계 인식에 미치는 영향을 규명

**핵심 질문**:
1. 불안 수준이 신경 사건 경계 탐지에 어떤 영향을 미치는가?
2. 다양한 event segmentation 방법론(BSDS, HMM) 간 비교 성능은?
3. 개인차(individual differences)가 뇌 상태 역학에 어떻게 반영되는가?

## 구현된 방법론

| Method | 특징 | Reference |
|--------|------|-----------|
| **BSDS** | Bayesian + AR(1) dynamics + Factor model | Taghia et al. 2018 Nature Comm |
| **HMM-Baldassano** | Event-sequential HMM | Baldassano et al. 2017 Neuron |
| **HMM-Yang** | Standard GaussianHMM | Yang et al. 2023 Nature Comm |

### 시뮬레이션 비교 결과

| Method | Avg F1 | Precision | Recall |
|--------|--------|-----------|--------|
| BSDS | 1.000 | 1.000 | 1.000 |
| HMM-Baldassano | 0.800 | 0.800 | 0.800 |
| HMM-Yang | 1.000 | 1.000 | 1.000 |

## 프로젝트 구조

```
UNSEAM/
├── BSDS_Project/              # Event segmentation 방법론
│   ├── bsds_complete/         # BSDS Python 구현
│   ├── hmm_baseline/          # HMM Baseline (Baldassano + Yang)
│   ├── compare_methods.py     # 방법 비교 스크립트
│   ├── scripts/               # SLURM job scripts
│   └── docs/                  # 상세 문서
│
├── analysis/                  # 분석 스크립트
│   ├── run_hmm_emofilm.py    # Emo-Film HMM 분석
│   └── test_hmm_boundary_detection.py
│
├── docs/                      # 연구 문서
│   ├── literature_review_event_boundaries_anxiety.md
│   ├── research_design.md
│   └── dataset_evaluation_*.md
│
├── daily_progress/            # 일일 진행 기록
│
├── data/                      # 데이터 (gitignored)
├── models/                    # 모델 체크포인트
├── results/                   # 분석 결과
└── papers/                    # 참고 논문
```

## Quick Start

### 설치

```bash
# 환경 생성
conda create -n unseam python=3.9
conda activate unseam

# 의존성 설치
pip install numpy scipy scikit-learn hmmlearn
pip install nilearn nibabel matplotlib seaborn
```

### 방법 비교 실행

```bash
cd BSDS_Project

# 시뮬레이션 데이터로 세 가지 방법 비교
python compare_methods.py --n-events 8

# BSDS 실행
python run_emofilm_bsds.py --task BigBuckBunny --n-states 8

# HMM Baseline 실행
python run_hmm_baseline.py --mode test
```

### 랩서버에서 실행

```bash
cd BSDS_Project
sbatch scripts/run_hmm_emofilm.slurm
```

## 연구 배경

### 핵심 발견 (문헌 고찰)

#### ✅ 신경-행동 경계 분리 (Baldassano et al., 2017)
- 신경 사건 경계는 행동 보고보다 **더 빈번하고 세밀함**
- 35-40% 일치도는 계층적 처리를 반영
- 초기 감각 영역: 매우 세밀한 경계
- DMN/고차 영역: 의식적 경계와 더 잘 대응

#### ✅ HMM의 타당성
- Gold-standard 방법으로 확립
- StudyForrest, Sherlock 등에서 광범위하게 검증
- 개인차 포착에 최적

#### 🔴 연구 공백: 불안과 신경 사건 분할
- **직접적 연구 전무**: 불안이 신경 사건 분할에 미치는 영향 미조사
- DMN 변화 + 개인차 연구는 각각 존재하나 통합 연구 없음
- **높은 novelty와 impact 예상**

## 데이터셋

| Dataset | 특징 | 불안 척도 |
|---------|------|-----------|
| Emo-FiLM | 감정 영화 시청 fMRI | ❌ |
| Sherlock | TV 시청 fMRI | ❌ |
| StudyForrest | Forrest Gump 시청 | ❌ |
| **신규 수집 필요** | 특성 불안 측정 포함 | ✅ 필요 |

## Documentation

- **HMM Baseline 상세**: `BSDS_Project/docs/HMM_BASELINE_MANUAL.md`
- **문헌 리뷰**: `docs/literature_review_event_boundaries_anxiety.md`
- **연구 설계**: `docs/research_design.md`
- **랩서버 가이드**: `analysis/LABSERVER_SETUP.md`

## References

### 방법론
1. **Taghia et al. (2018)** - BSDS
   - "Uncovering hidden brain state dynamics..."
   - *Nature Communications*, 9, 2505

2. **Baldassano et al. (2017)** - Event-sequential HMM
   - "Discovering event structure in continuous narrative..."
   - *Neuron*, 95(3), 709-721

3. **Yang et al. (2023)** - Standard GaussianHMM
   - "The default network dominates neural responses..."
   - *Nature Communications*, 14, 4400

### 이론적 배경
- Zacks et al. - Event Segmentation Theory
- Eysenck's Attentional Control Theory
- Bar-Haim et al. (2007) - Threat-related attentional bias

## License

MIT License

## Author

Kyungjin Oh (castella@snu.ac.kr)

---
*Last updated: 2026-01-14*
