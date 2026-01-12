# BSDS Python Port Critical Analysis Report

**Date**: 2025-12-15
**Analyst**: Claude Code
**Purpose**: MATLAB → Python 이식 검증 및 emo-film 적용 가이드

---

## 1. Executive Summary

현재 Python 포팅은 **핵심 추론 단계의 약 60-70%가 누락**되어 있습니다. HMM Forward-Backward와 초기화 부분은 구현되어 있으나, BSDS의 핵심인 AR-FA (Autoregressive Factor Analysis) 학습 루프가 대부분 미구현 상태입니다.

**결론**: 현재 상태로는 신뢰할 수 있는 결과를 얻기 어렵습니다. 핵심 함수들의 구현이 필요합니다.

---

## 2. Detailed Comparison

### 2.1 구현된 부분 (Implemented) ✅

| Component | MATLAB File | Python File | Status |
|-----------|-------------|-------------|--------|
| Main Class | `BayesianSwitchingDynamicalSystems.m` | `bsds_model.py` | ⚠️ 부분 구현 |
| HMM E-Step | `vbhmmEstep.m` | `bsds_inference.py:vbhmm_estep()` | ✅ 구현됨 |
| Forward-Backward | `VBHMMforward.m`, `VBHMMbackward.m` | `bsds_inference.py` (inline) | ✅ 구현됨 |
| K-Means Init | `initPoteriors.m` | `bsds_utils.py:init_posteriors_kmeans()` | ✅ 구현됨 |
| Log Output Probs | `computeLogOutProbs.m` | `bsds_inference.py:compute_log_out_probs()` | ⚠️ 부분 구현 |
| ARD Update | `inferQnu.m` | `bsds_learning.py:infer_q_nu()` | ⚠️ 부분 구현 |
| KL Dirichlet | `kldirichlet.m` | `bsds_utils.py:kl_dirichlet()` | ✅ 구현됨 |

### 2.2 누락된 핵심 부분 (Missing) ❌

| Component | MATLAB File | Importance | Description |
|-----------|-------------|------------|-------------|
| **AR(1) Inference** | `inferAR3.m` | 🔴 Critical | AR 동역학 학습 - BSDS 핵심 |
| **Factor Loading Update** | `inferQL.m` | 🔴 Critical | L 행렬 업데이트 |
| **Latent Variable Update** | `inferQX.m` | 🔴 Critical | X 잠재변수 추론 |
| **Noise Variance** | `inferpsii2.m` | 🟡 Important | 관측 노이즈 학습 |
| **Mean Update** | `infermcl.m` | 🟡 Important | 평균 파라미터 |
| **VAR M-Step** | `mstep_VBVAR.m` | 🔴 Critical | VAR 계수 업데이트 |
| **Lower Bound** | `computeLowerBound.m` | 🟡 Important | ELBO 수렴 모니터링 |
| **Viterbi Decoding** | `estimateStatesByVitterbi.m` | 🟡 Important | 최적 상태 시퀀스 |
| **Post-hoc Stats** | `compute_occupancy_and_mean_life_*.m` | 🟢 Optional | 결과 분석 |

### 2.3 수학적 오류 분석

#### 2.3.1 `compute_log_out_probs` 비교

**MATLAB (원본)**:
```matlab
logQns(:,col) = -.5*( +sum(Y.*(diag(psii)*(Y-2*L*X)),1)' ...
    +trace(temp) * trace(Xcov) ...
    +sum(X.*(temp*X),1)' ...
    +trace(Xcov(2:end,2:end)) ...
    +sum(X(2:end,:).*X(2:end,:),1)' ...
    -2*sum(log(diag(chol(Xcov(2:end,2:end))))) );
```

**Python (현재)**:
```python
# term_b 계산에서 Xcov의 모든 시점 합산 문제
term_b = np.einsum('ij,ijt->t', temp_mat, Xcov_subj)  # 인덱싱 불일치
```

**문제점**:
- MATLAB은 피험자별로 분리된 `Xcov`를 사용
- Python은 전역 `Xcov` 배열에서 잘못된 슬라이싱

#### 2.3.2 `learnAR_FA` 루프 부재

**MATLAB 메인 루프** (`vbhafa.m:113-133`):
```matlab
while iter<=nIter && improvement>tol
    learnAR_FA;  % 이 호출이 핵심!
    computeLowerBound;
    ...
end
```

**Python 메인 루프** (`bsds_model.py:39-55`):
```python
for it in range(self.n_iter):
    log_emissions = infer.compute_log_out_probs(...)
    # learnAR_FA 호출 없음 ← 핵심 누락
    wa_new, wpi_new = learn.update_transition_counts(...)
```

---

## 3. Impact Assessment

### 3.1 현재 코드가 수행하는 것
1. K-Means로 초기 상태 할당
2. HMM Forward-Backward로 γ (감마) 계산
3. 전이 확률 업데이트

### 3.2 현재 코드가 수행하지 못하는 것
1. ❌ AR(1) 동역학 학습 (상태 전환 패턴)
2. ❌ Factor Loading 학습 (차원 축소)
3. ❌ 잠재 변수 X 업데이트 (시간에 따른 진화)
4. ❌ 관측 노이즈 추정
5. ❌ ELBO 기반 수렴 판단
6. ❌ Viterbi 최적 경로 추출

**결과**: 출력되는 상태 시퀀스는 단순 K-Means + HMM smoothing 수준이며, BSDS의 핵심인 동적 시스템 모델링이 없습니다.

---

## 4. Recommendations

### 4.1 즉시 필요한 수정 (Critical)

```
우선순위 1: inferQX.m → bsds_inference.py
우선순위 2: inferQL.m → bsds_learning.py
우선순위 3: inferAR3.m + mstep_VBVAR.m → bsds_learning.py
우선순위 4: inferpsii2.m, infermcl.m → bsds_learning.py
```

### 4.2 대안적 접근법

만약 시간이 촉박하다면:

1. **MATLAB 직접 사용**: 원본 MATLAB 코드가 완전히 작동하므로, MATLAB Runtime이나 Octave 활용
2. **기존 Python 라이브러리**: `hmmlearn`, `pyhsmm` 등 검증된 라이브러리 사용 후 AR 확장
3. **부분 구현**: HMM 부분만 사용하고 AR 동역학은 별도 분석

### 4.3 테스트 전략

구현 후 검증:
```python
# MATLAB .mat 결과와 Python 결과 비교
from scipy.io import loadmat
matlab_result = loadmat('test_result.mat')
python_result = model.fit(data)
assert np.allclose(matlab_result['stran'], python_result.stran, atol=1e-4)
```

---

## 5. Emo-Film 데이터 적용 가이드

### 5.1 데이터 구조
```
Emo-FilM/
├── derivatives/preprocessing/
│   └── sub-S*/ses-1/func/
│       └── sub-S*_ses-1_task-{Movie}_space-MNI_desc-ppres_bold.nii.gz
```

### 5.2 권장 파이프라인

```python
# Step 1: ROI 추출 (현재 run_extraction.py 사용)
python run_extraction.py sub-S01

# Step 2: 데이터 준비
import numpy as np
ts = np.load('sub-S01_timeseries.npy')  # (Time x 400 ROI)
data = ts.T  # (400 x Time) - BSDS 입력 형식

# Step 3: BSDS 피팅 (수정된 구현 필요)
from bsds import BSDSModel
model = BSDSModel(n_states=5, max_ldim=10)
model.fit([data])

# Step 4: 결과 분석
states = model.get_viterbi_path()
occupancy = model.compute_occupancy()
```

### 5.3 권장 파라미터

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_states | 5-8 | fMRI 연구 표준 |
| max_ldim | 10-20 | 400 ROI → 차원 축소 |
| n_iter | 100 | 수렴 보장 |
| tol | 1e-3 | 표준 수렴 기준 |

---

## 6. 결론

현재 Python 포팅은 **프로토타입 수준**으로, 연구에 사용하기 전 핵심 추론 함수들의 완전한 구현이 필요합니다. MATLAB 원본 코드는 완전하게 작동하므로, 시간이 촉박한 경우 MATLAB 사용을 권장합니다.

**다음 단계**:
1. 누락된 함수 구현 (예상 시간: 1-2일)
2. MATLAB 결과와 교차 검증
3. Emo-Film 데이터 분석 진행

---

*Generated by Claude Code - 2025-12-15*
