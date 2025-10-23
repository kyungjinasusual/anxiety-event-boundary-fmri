# 연구 설계: Event Boundary Detection in Trait/State Anxiety using fMRI Transformer

**작성일**: 2024-10-23
**연구자**: Kyungjin Oh
**소속**: Seoul National University

---

## 1. 연구 배경 및 목적

### 1.1 연구 배경

**Event Segmentation Theory (EST)**는 인간이 연속적인 경험을 의미 있는 단위로 분할한다고 제안합니다 (Zacks et al., 2007). 사건 경계(event boundary)는:
- 예측 오류가 발생하는 시점
- 시간적 맥락이 리셋되는 순간
- 에피소드 기억 공고화의 중요 시점

**불안(Anxiety)**은:
- 예측 처리 이상과 관련 (intolerance of uncertainty)
- 편도체-전전두엽 회로 변화
- Default Mode Network (DMN) 역학 이상

**Research Gap**: 불안이 사건 경계 탐지에 미치는 영향은 미탐구 영역입니다.

### 1.2 연구 목적

본 연구는 **resting-state fMRI**에서 **fMRI Transformer (SwiFT)**를 활용하여:

1. 특성 불안(trait anxiety)과 상태 불안(state anxiety)이 사건 경계 탐지에 미치는 영향 규명
2. Resting-state에서 내재적 사건 구조 탐지
3. Transformer attention mechanism의 신경과학적 해석
4. 임상적으로 유용한 불안 신경 바이오마커 개발

---

## 2. 연구 가설

### **주 가설 (Primary Hypothesis)**

**H1**: 특성 불안 수준이 높은 개인은 resting-state fMRI에서 더 많은 사건 경계를 탐지할 것이다.

**근거**:
- 불안 ↑ → 불확실성 불내성 ↑
- 예측 오류 민감도 증가
- 사건 경계 과잉 탐지 (hypervigilance)

**통계 검증**:
```
Pearson correlation: STAI-T score × event boundary count
Expected: r > 0.3, p < 0.05
```

### **부 가설 (Secondary Hypotheses)**

**H2**: 상태 불안이 높을 때 사건 경계 강도(boundary strength)가 증가할 것이다.

**H3**: 불안 수준에 따라 SwiFT attention weights의 패턴이 다를 것이다.
- 고불안: 편도체-전전두엽 attention ↑
- 저불안: DMN 내부 attention ↑

**H4**: 사건 경계에서 해마-후방내측망(PMN) 연결성이 불안 수준과 부적 상관을 보일 것이다.
- 고불안 → 기억 공고화 효율성 ↓

---

## 3. 연구 설계

### 3.1 연구 유형
- **Observational, Cross-sectional Study**
- Between-group comparison + Dimensional analysis (correlation)

### 3.2 참가자

#### **표본 크기 (Sample Size)**

**Power analysis**:
- Effect size: r = 0.35 (medium effect, 선행 연구 기반)
- α = 0.05 (two-tailed)
- Power (1-β) = 0.80
- **Required N = 64**

**최종 목표**: N = 80 (dropout 20% 고려)

#### **포함 기준 (Inclusion Criteria)**

1. 연령: 만 20-40세
2. 오른손잡이
3. 한국어 모국어 사용자
4. 정상 또는 교정시력 정상
5. MRI 스캐닝 가능 (금속 물질 없음)
6. 연구 참여 동의서 서명

#### **제외 기준 (Exclusion Criteria)**

1. 현재 정신과 약물 복용 중
2. 신경학적 질환 병력
3. 뇌 손상 병력
4. 약물/알코올 의존 병력
5. MRI 금기증 (claustrophobia, 임신 등)
6. 시각/청각 장애
7. 최근 6개월 내 항우울제/항불안제 복용

#### **그룹 분류**

**방법 1: 연속 변인 분석** (Primary)
- STAI-T score를 연속 변인으로 사용
- Correlation & regression analysis

**방법 2: 그룹 비교** (Secondary)
- 고불안 그룹 (High Anxiety): STAI-T ≥ 50 (상위 33%)
- 중불안 그룹 (Moderate): STAI-T 40-49
- 저불안 그룹 (Low Anxiety): STAI-T < 40 (하위 33%)

---

## 4. 측정 도구

### 4.1 불안 측정

#### **STAI (State-Trait Anxiety Inventory)**

**특성 불안 (STAI-T)**:
- 20 문항
- 4점 척도 (거의 그렇지 않다 ~ 거의 언제나 그렇다)
- 점수 범위: 20-80
- 시점: fMRI 스캔 전

**상태 불안 (STAI-S)**:
- 20 문항
- 4점 척도
- 시점:
  - fMRI 스캔 직전 (pre-scan)
  - fMRI 스캔 직후 (post-scan)

#### **추가 척도** (탐색적)

**IUS (Intolerance of Uncertainty Scale)**:
- 불확실성 불내성 측정
- 27 문항, 5점 척도

**BDI-II (Beck Depression Inventory)**:
- 우울 수준 통제 변인
- 21 문항

### 4.2 fMRI 데이터 수집

#### **Resting-state fMRI Protocol**

**스캐너**: 3T MRI scanner (Siemens/GE/Philips)

**Structural (T1-weighted)**:
- TR/TE: 2000/2.5 ms
- Voxel size: 1×1×1 mm³
- Slices: 176 sagittal

**Functional (EPI)**:
- TR: 2000 ms (권장) 또는 720 ms (multiband)
- TE: 30 ms
- Flip angle: 90°
- Voxel size: 3×3×3 mm³
- Slices: 36-40 (whole brain)
- Volumes: 300 (10분) 또는 600 (20분 추천)
- Eyes: Open (fixation cross) 또는 Closed

**Instruction**:
```
"편안히 눕고 눈을 감으세요 (또는 십자가를 보세요).
특별히 생각하려고 노력하지 마시고, 자연스럽게 두세요.
움직이지 마세요."
```

#### **Field Map** (선택적, 권장)
- Distortion correction용

---

## 5. 데이터 분석 파이프라인

### 5.1 전처리 (Preprocessing)

**도구**: fMRIPrep 또는 SPM12

**단계**:
1. Slice timing correction
2. Motion correction (realignment)
3. Coregistration (T1 ↔ EPI)
4. Normalization (MNI space)
5. Spatial smoothing (6mm FWHM)
6. Temporal filtering (0.01-0.1 Hz bandpass)
7. Nuisance regression:
   - 6 motion parameters + derivatives
   - CSF & WM signals
   - Global signal (optional, controversial)

**Quality Control**:
- Framewise displacement (FD) < 0.5 mm
- DVARS threshold
- Exclude volumes with FD > 0.5 mm (scrubbing)

### 5.2 Event Boundary Detection

#### **Method 1: SwiFT (Primary)**

**모델**: Swin 4D fMRI Transformer
**GitHub**: https://github.com/athms/swift-fmri

**절차**:
1. Pre-trained SwiFT 로드
2. Fine-tuning for event boundary detection:
   - Self-supervised: temporal contrastive learning
   - Objective: Detect state transitions
3. Inference:
   - Input: 4D fMRI volume (time × x × y × z)
   - Output: Event boundary probability per TR
4. Threshold: Top 10% or adaptive threshold

**Attention Analysis**:
- Multi-head attention weights 추출
- Spatial attention maps 시각화
- Temporal attention patterns 분석

#### **Method 2: Baselines**

**Hidden Markov Model (HMM)**:
- States: 3-10 (cross-validation)
- Features: Regional mean signals (AAL atlas)
- Transition probabilities → boundary detection

**Sliding Window Correlation**:
- Window size: 30-60 seconds
- Step: 1 TR
- Correlation change → boundary

**GSBS (Greedy State Boundary Search)**:
- Boundary detection algorithm (Geerligs et al., 2021)

### 5.3 Connectivity Analysis (Traditional)

**목적**: SwiFT attention의 신경과학적 검증

**ROI Selection**:
- Amygdala (편도체)
- vmPFC (ventromedial prefrontal cortex)
- dmPFC (dorsomedial prefrontal cortex)
- PCC (posterior cingulate cortex)
- Hippocampus (posterior)
- Angular gyrus

**Networks**:
- DMN (Default Mode Network)
- SN (Salience Network)
- CEN (Central Executive Network)

**Analysis**:
1. **Seed-based correlation**:
   - Seed: Amygdala
   - Target: whole brain
   - Contrast: High vs Low anxiety

2. **Dynamic Functional Connectivity (dFC)**:
   - Sliding window (30-60s)
   - Correlation at event boundaries vs non-boundaries

3. **Network Switching**:
   - DMN ↔ SN transitions
   - Frequency at event boundaries

### 5.4 Hybrid Interpretation

**Attention → Connectivity Mapping**:
```python
# Pseudo-code
attention_weights = swift_model.get_attention(fmri_data)
roi_attention = extract_roi_attention(attention_weights, roi_masks)
connectivity_matrix = compute_correlation(roi_attention)

# Compare with traditional connectivity
traditional_conn = compute_seed_correlation(fmri_data)
correlation(roi_attention, traditional_conn)
```

**Validation**:
- Attention weights와 seed-based connectivity 상관
- Expected: r > 0.5

---

## 6. 통계 분석

### 6.1 Primary Analysis

**연구 질문 1**: 특성 불안과 사건 경계 개수

**분석**:
```r
# Correlation
cor.test(STAI_T, event_boundary_count, method="pearson")

# Multiple regression (control for confounds)
lm(event_boundary_count ~ STAI_T + age + sex + motion + BDI)
```

**예상 결과**:
- r(STAI_T, boundary_count) > 0.3, p < 0.05

### 6.2 Secondary Analyses

**연구 질문 2**: 상태 불안과 경계 강도

```r
# Mixed-effects model
lmer(boundary_strength ~ STAI_S + (1|subject))
```

**연구 질문 3**: 그룹 비교

```r
# ANOVA
aov(boundary_count ~ anxiety_group)
# Post-hoc: Tukey HSD
```

**연구 질문 4**: Attention patterns

```python
# Permutation test
from scipy.stats import permutation_test

def statistic(high_anxiety_attn, low_anxiety_attn):
    return np.mean(high_anxiety_attn) - np.mean(low_anxiety_attn)

result = permutation_test(
    (high_attn, low_attn),
    statistic,
    n_resamples=10000
)
```

### 6.3 Confound Control

**통제 변인**:
- Age (연령)
- Sex (성별)
- Education (교육 수준)
- Head motion (FD mean)
- Depression (BDI score)

**방법**: Partial correlation, Multiple regression

### 6.4 Multiple Comparison Correction

**Behavioral**: Bonferroni correction (α = 0.05/n)

**Neuroimaging**:
- Cluster-level FWE correction (p < 0.05)
- FDR correction for ROI analyses

---

## 7. 예상 결과

### 7.1 행동 결과

**Table 1: Participant Characteristics**
| Variable | High Anxiety (n=27) | Low Anxiety (n=27) | p-value |
|----------|---------------------|-------------------|---------|
| Age | 26.3 ± 4.2 | 25.8 ± 3.9 | 0.65 |
| Sex (F/M) | 18/9 | 16/11 | 0.58 |
| STAI-T | 53.2 ± 3.1 | 35.4 ± 4.2 | <0.001 |
| STAI-S (pre) | 48.1 ± 5.3 | 32.6 ± 4.8 | <0.001 |
| BDI-II | 12.3 ± 6.2 | 5.4 ± 3.1 | <0.001 |
| Event Boundary Count | 28.5 ± 6.3 | 21.2 ± 4.7 | <0.01 |

**Figure 1**: Scatterplot STAI-T × Event Boundary Count
- Positive correlation: r = 0.42, p < 0.001

### 7.2 신경영상 결과

**Figure 2**: SwiFT Event Boundary Detection
- Panel A: Example event boundary timeline
- Panel B: Boundary strength distribution (High vs Low anxiety)
- Panel C: Attention maps at boundaries

**Figure 3**: Attention Patterns
- Panel A: Amygdala-vmPFC attention (High > Low)
- Panel B: DMN internal attention (Low > High)
- Panel C: Network switching frequency

**Figure 4**: Connectivity Validation
- Panel A: Seed-based (Amygdala) connectivity maps
- Panel B: Correlation: Attention weights ↔ Connectivity
- Panel C: Dynamic FC at event boundaries

### 7.3 Hybrid Interpretation

**Table 2: ROI Attention vs Connectivity**
| ROI Pair | Attention Weight | Seed Correlation | r | p |
|----------|-----------------|------------------|---|---|
| Amy-vmPFC | 0.62 ± 0.12 | 0.58 ± 0.15 | 0.71 | <0.001 |
| PCC-Angular | 0.54 ± 0.10 | 0.49 ± 0.12 | 0.68 | <0.001 |

---

## 8. 타임라인

### **Phase 1: 준비 (Week 1-4)**
- [ ] IRB 승인
- [ ] 참가자 모집 광고
- [ ] SwiFT 구현 및 테스트
- [ ] Baseline 모델 구현

### **Phase 2: 파일럿 (Week 5-6)**
- [ ] 파일럿 참가자 5명 스캔
- [ ] 프로토콜 최적화
- [ ] 분석 파이프라인 검증

### **Phase 3: 데이터 수집 (Week 7-18)**
- [ ] 80명 fMRI 스캔 (주당 6-7명)
- [ ] Quality control (매주)
- [ ] 데이터 백업

### **Phase 4: 분석 (Week 19-24)**
- [ ] 전처리 (fMRIPrep)
- [ ] SwiFT event boundary detection
- [ ] Baseline 비교
- [ ] Connectivity analysis
- [ ] 통계 분석

### **Phase 5: 논문 작성 (Week 25-30)**
- [ ] 초안 작성
- [ ] 그림/표 제작
- [ ] 내부 리뷰
- [ ] 투고 준비

### **Phase 6: 투고 및 리뷰 (Week 31+)**
- [ ] 저널 투고
- [ ] 리뷰 대응
- [ ] 최종 출판

**Total**: ~7-8개월

---

## 9. 예산 (개략)

| 항목 | 단가 | 수량 | 합계 |
|------|------|------|------|
| fMRI 스캔 | 300,000원 | 80명 × 1시간 | 24,000,000원 |
| 참가자 사례비 | 50,000원 | 80명 | 4,000,000원 |
| 컴퓨팅 (GPU) | 2,000,000원 | 6개월 | 2,000,000원 |
| 기타 (소모품) | - | - | 1,000,000원 |
| **총계** | | | **31,000,000원** |

---

## 10. 윤리적 고려사항

### 10.1 IRB 승인
- 서울대학교 IRB 신청
- 참가자 동의서 준비 (한국어)

### 10.2 참가자 보호
- MRI 안전 스크리닝
- 불안 악화 모니터링
- 필요 시 상담 연계

### 10.3 데이터 보안
- 개인정보 비식별화
- 암호화된 서버 저장
- 3년 후 파기 (또는 참가자 동의 시 연구 목적 보관)

---

## 11. 예상되는 기여

### 11.1 이론적 기여
- Event Segmentation Theory를 임상 신경과학으로 확장
- 불안의 예측 처리 이상 메커니즘 규명

### 11.2 방법론적 기여
- Resting-state에서 사건 경계 탐지 (최초)
- fMRI Transformer 적용 (새로운 분석법)
- Hybrid interpretation framework

### 11.3 임상적 기여
- 불안 신경 바이오마커 개발
- 치료 타겟 제시 (DMN-SN switching)
- 개인 맞춤형 치료 가능성

---

## 12. 한계점 및 대응

### 한계점 1: Resting-state의 ground truth 부재
**대응**:
- Multiple methods convergence (SwiFT + HMM + GSBS)
- Reproducibility test (split-half reliability)

### 한계점 2: SwiFT interpretability
**대응**:
- Hybrid approach (전통적 connectivity로 검증)
- Attention visualization
- Perturbation analysis

### 한계점 3: 단면 연구 (causality 불가)
**대응**:
- 상관 연구로 명시
- 향후 종단 연구 또는 실험 연구 제안

### 한계점 4: 샘플의 대표성
**대응**:
- 다양한 불안 수준 포함 (연속 변인)
- 임상 집단 추가 모집 고려

---

## 13. 참고문헌

1. Zacks et al. (2007). Event perception: A mind-brain perspective. *Psychological Bulletin*.
2. Barnett et al. (2024). Hippocampal-cortical interactions. *Neuron*.
3. Heusser et al. (2021). Default mode network in naturalistic perception. *Communications Biology*.
4. Kumar et al. (2023). Bayesian surprise. *Cognitive Science*.

[전체 참고문헌은 literature_review_2024.md 참조]

---

**문서 버전**: 1.0
**최종 수정**: 2024-10-23
**다음 업데이트**: 파일럿 데이터 수집 후
