# UNSEAM

**U**ncertainty and **S**urprise in **E**vent boundary **A**nxiety **M**odulation

*"How we weave a story with small pieces of events"*

불안(anxiety)이 신경 사건 분할(neural event segmentation)에 미치는 영향을 규명하는 연구 프로젝트

## 연구 동기

> "People don't see the world before their eyes until it's put in a narrative mode."
> — Filmmaker Brian De Palma

인간의 마음은 서사 생성 엔진(narrative generation engine)으로서, 연속적인 경험을 이산적인 사건 단위로 분할하여 의미를 구성합니다. 이 연구는 불안이 이러한 사건 분할 과정에 어떤 영향을 미치는지 탐구합니다.

## 이론적 배경

### Event Segmentation Theory (EST)
- 연속적 경험을 이산적, 의미 있는 사건 단위로 자동 분할 (Zacks et al., 2007)
- **핵심 원리**: 사건 경계는 예측 오차(prediction error)가 일시적으로 증가할 때 발생
- GPT-2 Bayesian surprise가 인간 사건 분할과 유의미하게 상관 (r ≈ 0.3-0.4, Kumar et al., 2023)

### 사건 경계의 계층적 구조
| Level | Brain Regions | Event Type | Example |
|-------|---------------|------------|---------|
| Sensory | V1, A1 | Short events | Screen cuts, music changes |
| Action | STS | Medium events | Handshakes, car departures |
| Situation | AG, PMC | Long events | Job interviews, breakups |

고차원 사건 경계에서 해마 결합(hippocampal binding) 현상이 관찰되며, 이는 일화 기억 형성과 직결됩니다.

### Uncertainty and Anticipation Model of Anxiety (UAMA)
불안은 예측 처리(predictive processing)를 변형시킵니다 (Grupe & Nitschke, 2013):
- 과잉경계(hypervigilance): 뇌의 신호 탐지 민감도 비정상적 증가
- 일반인이 무시할 작은 감각/맥락 변화도 불안한 뇌에서는 예측 오차 유발

## 연구 가설

### 핵심 질문
> 불안한 개인은 세계를 다르게 분할하는가?

### 상충하는 선행 연구

| (A) Fuzzy Boundaries | (B) Clear-cut Boundaries |
|---------------------|-------------------------|
| "불안은 경계를 흐리게 한다" | "불안은 경계를 과도하게 정밀하게 한다" |
| 분할 능력 손상, 과일반화 | 안전 학습 방해, 과분할 |
| Cooper et al., 2022; Duits et al., 2015 | Zika et al., 2023; Gershman et al., 2010 |

### 본 연구의 가설 (Supporting B)
- 불안은 **과분할(hypersegmentation)**을 유발
- 높은 특성 불안 → 더 빈번한 신경 사건 경계, 더 짧은 상태 체류 시간
- 불안한 개인은 사건을 "잘" 분할하지만, 너무 정밀하고 경직된 방식으로 수행

## 방법론

### 순차적 확장 구조

본 연구는 세 단계의 방법론적 확장을 통해 신경 사건 경계 탐지의 정밀도와 해석력을 점진적으로 향상시킵니다:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 1: HMM Baseline        →  Phase 2: BSDS           →  Phase 3: Deep Learning  │
│  ─────────────────────           ─────────────               ──────────────────────  │
│  • Event-sequential HMM          • Bayesian inference        • GST-UNet              │
│  • Standard GaussianHMM          • AR(1) dynamics            • SWIFT                 │
│  • 빠른 구현, 검증된 방법           • Factor model              • NeuroMamba            │
│                                  • 자동 상태 수 결정            • Causal inference      │
│                                                              • Future simulation     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 구현된 방법

| Phase | Method | 특징 | Status |
|-------|--------|------|--------|
| **1** | HMM-Baldassano | Event-sequential HMM, 이산적 뇌 상태 가정 | ✅ 구현 완료 |
| **1** | HMM-Yang | Standard GaussianHMM, hmmlearn 기반 | ✅ 구현 완료 |
| **2** | BSDS | Bayesian + AR(1) dynamics + Factor model | ✅ 구현 완료 |
| **3** | GST-UNet | Spatiotemporal causal inference | 🔄 계획 중 |
| **3** | SWIFT/NeuroMamba | State-of-the-art neural sequence models | 🔄 검토 중 |

### 방법별 비교

| 측면 | BSDS | HMM-Baldassano | HMM-Yang |
|------|------|----------------|----------|
| **상태 전이** | 자유 | Sequential only | 자유 |
| **Emission** | Factor model + AR(1) | Gaussian | Gaussian |
| **상태 수 결정** | 자동 (Bayesian) | 사전 지정 필요 | BIC 기반 선택 |
| **시간 역학** | AR(1) dynamics | 없음 | 없음 |

### 시뮬레이션 비교 결과

| Method | Avg F1 | Precision | Recall |
|--------|--------|-----------|--------|
| BSDS | 1.000 | 1.000 | 1.000 |
| HMM-Baldassano | 0.800 | 0.800 | 0.800 |
| HMM-Yang | 1.000 | 1.000 | 1.000 |

## 데이터셋

### Primary: Emo-FilM

Morgenroth et al. (2025)의 자연주의 영화 시청 fMRI 데이터셋:

| 항목 | 내용 |
|------|------|
| **참가자** | 30명 (여성 18명, 평균 연령 25.83세, 건강군) |
| **자극** | 14편 단편 영화 (평균 11분 26초, 총 2.5시간+) |
| **fMRI** | 3T, TR=1.3초 |
| **감정 annotation** | 44명 평가자의 1Hz 연속 감정 평정 |
| **불안 측정** | DASS-21 불안 하위 척도 |

### 추가 데이터셋 (계획)

| Dataset | 특징 | 불안 척도 | Status |
|---------|------|----------|--------|
| **Emo-FilM** | 감정 영화 14편, 연속 감정 annotation | ✅ DASS-21 | 🔄 진행 중 |
| **Sherlock** | TV 시청 fMRI, HMM 검증 표준 | ❌ | 📋 검토 중 |
| **StudyForrest** | Forrest Gump 시청, 풍부한 annotation | ❌ | 📋 검토 중 |
| **HCP Movie** | 대규모 표준화 데이터 | ❌ | 📋 검토 중 |
| **신규 수집** | 특성 불안 측정 포함 설계 | ✅ 필요 | 📋 계획 중 |

### 주요 변수
- **독립변수**: DASS-21 불안 점수
- **종속변수**:
  - 신경 사건 경계 빈도 (HMM/BSDS 상태 전이 수)
  - 신경 사건 경계 집단 정렬도 (개인 vs. 집단 평균 경계 상관)

### 관심 영역 (ROIs)
- V1/A1 (감각 수준)
- Superior Temporal Sulcus (STS, 행동 수준)
- Angular Gyrus (AG, 상황 수준)
- Posterior Medial Cortex (PMC: PCC, precuneus, retrosplenial)
- Hippocampus (HPC)

## 프로젝트 구조

```
UNSEAM/
├── BSDS_Project/                    # 🧠 Event Segmentation Methods
│   │
│   ├── hmm_baseline/                # Phase 1: HMM Baseline
│   │   ├── model.py                 #   - Baldassano 스타일 Event-sequential HMM
│   │   ├── hmmlearn_wrapper.py      #   - Yang 스타일 Standard GaussianHMM
│   │   └── comparison.py            #   - 방법 비교 유틸리티
│   │
│   ├── bsds_complete/               # Phase 2: BSDS
│   │   ├── core/                    #   - BSDSModel, BSDSConfig
│   │   ├── inference/               #   - HMM, Latent variable inference
│   │   └── learning/                #   - Factor, AR, Transition learning
│   │
│   ├── deep_learning/               # Phase 3: Deep Learning (계획)
│   │   ├── gst_unet/                #   - GST-UNet implementation
│   │   └── neural_seq/              #   - SWIFT, NeuroMamba 등
│   │
│   ├── compare_methods.py           # 방법 비교 스크립트
│   ├── scripts/                     # SLURM job scripts
│   └── docs/                        # 상세 문서
│
├── analysis/                        # 분석 스크립트
├── docs/                            # 연구 문서
├── daily_progress/                  # 일일 진행 기록
│
├── data/                            # 데이터 (gitignored)
├── models/                          # 모델 체크포인트
├── results/                         # 분석 결과
└── papers/                          # 참고 논문
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

# HMM Baseline 테스트
python run_hmm_baseline.py --mode test

# Emo-Film 데이터로 실행
python run_hmm_baseline.py --dataset emofilm --task BigBuckBunny --n-events 8
```

### 랩서버에서 실행

```bash
cd BSDS_Project
sbatch scripts/run_hmm_emofilm.slurm
```

## 연구의 의의

| 영역 | 기대 효과 |
|------|----------|
| **불안의 인지 메커니즘** | 불안은 단순한 "걱정"이 아닌, 세계를 분할하고 인식하는 방식의 근본적 변화일 수 있음 |
| **불안 유지 기제 설명** | 과분할이 부정적 기억의 과도한 분리 및 침습과 연결될 가능성 |
| **임상적 적용** | 불안 장애에 대한 시사점, 특정 치료법의 효과 이유 규명 |

## Documentation

- **HMM Baseline 상세**: `BSDS_Project/docs/HMM_BASELINE_MANUAL.md`
- **문헌 리뷰**: `docs/literature_review_event_boundaries_anxiety.md`
- **연구 설계**: `docs/research_design.md`
- **랩서버 가이드**: `BSDS_Project/scripts/README.md`

## References

### 핵심 방법론
1. **Baldassano et al. (2017)** - Event-sequential HMM
   - "Discovering event structure in continuous narrative perception and memory"
   - *Neuron*, 95(3), 709-721

2. **Taghia et al. (2018)** - BSDS
   - "Uncovering hidden brain state dynamics that regulate performance and decision-making during cognition"
   - *Nature Communications*, 9, 2505

3. **Yang et al. (2023)** - Standard GaussianHMM
   - "The default network dominates neural responses to evolving movie stories"
   - *Nature Communications*, 14, 4400

### 이론적 배경
- **Zacks et al. (2007)** - Event Segmentation Theory, *Psychological Bulletin*
- **Grupe & Nitschke (2013)** - UAMA, *Nature Reviews Neuroscience*
- **Kumar et al. (2023)** - Bayesian Surprise & Event Segmentation, *Cognitive Science*

### 데이터셋
- **Morgenroth et al. (2025)** - Emo-FilM, *Scientific Data*

## License

MIT License

## Author

Kyungjin Oh (castella@snu.ac.kr)
Master's Student, Department of Psychology
Connectome Lab, Seoul National University

---
*Last updated: 2026-01-14*
