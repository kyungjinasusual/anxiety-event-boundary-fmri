# Open Datasets for Anxiety & Resting-state fMRI Research

**작성일**: 2024-10-23
**목적**: 특성/상태 불안과 resting-state fMRI 데이터가 포함된 공개 데이터셋 정리

---

## 1. 추천 데이터셋

### **1.1 Human Connectome Project (HCP)**

**URL**: https://www.humanconnectome.org/

**데이터 규모**:
- N = 1,200 participants (young adults, 22-35세)
- High-quality 3T and 7T fMRI data

**포함 데이터**:
- ✅ Resting-state fMRI (4 runs, 15min each)
- ✅ Structural MRI (T1, T2)
- ✅ Behavioral/psychological assessments
- ⚠️ 불안 척도: 직접적 STAI는 없지만 대체 가능

**장점**:
- 고품질 데이터 (multiband, high spatial/temporal resolution)
- 잘 정리된 전처리 파이프라인
- 대규모 샘플

**단점**:
- 특성 불안 직접 측정치 없음 (DSM Anxiety 관련 변인은 있음)
- 정상 성인 중심 (임상 집단 부족)

**접근 방법**:
- HCP 계정 생성 (무료)
- Data Use Terms 동의
- AWS S3 또는 Aspera 다운로드

---

### **1.2 UK Biobank**

**URL**: https://www.ukbiobank.ac.uk/

**데이터 규모**:
- N = 500,000+ participants
- Imaging subset: ~100,000 (진행 중)

**포함 데이터**:
- ✅ Resting-state fMRI
- ✅ Task fMRI
- ✅ Mental health questionnaires (불안, 우울 등)
- ✅ Longitudinal data

**장점**:
- 매우 큰 샘플 사이즈
- 불안/우울 관련 상세 평가
- 종단 데이터
- 임상적 다양성

**단점**:
- 접근 신청 필요 (연구 계획서 제출)
- 승인 시간 소요 (수주~수개월)
- 데이터 다운로드 크기 매우 큼

**접근 방법**:
- UK Biobank Access Management System (AMS) 신청
- 연구 프로포절 제출
- 승인 후 데이터 접근

---

### **1.3 OpenNeuro**

**URL**: https://openneuro.org/

**추천 데이터셋**:

#### **ds000030 - UCLA Consortium for Neuropsychiatric Phenomics**
- N = 265 (includes anxiety disorders)
- Resting-state fMRI
- Clinical assessments
- **Dataset**: https://openneuro.org/datasets/ds000030

#### **ds002748 - Social Anxiety fMRI**
- N = 70 (social anxiety + controls)
- Resting-state + task fMRI
- STAI included
- **Dataset**: https://openneuro.org/datasets/ds002748

#### **ds003097 - Anxiety and Depression**
- N = 60
- Resting-state fMRI
- Anxiety/depression scales
- **Dataset**: https://openneuro.org/datasets/ds003097

**장점**:
- 즉시 다운로드 가능 (공개)
- BIDS 형식
- 불안 직접 측정

**단점**:
- 샘플 사이즈 작음
- 데이터 품질 다양

**접근 방법**:
- 웹사이트에서 직접 다운로드
- AWS S3 또는 DataLad

---

### **1.4 ABIDE (Autism Brain Imaging Data Exchange)**

**URL**: http://fcon_1000.projects.nitrc.org/indi/abide/

**데이터 규모**:
- ABIDE-I: N = 1,112
- ABIDE-II: N = 1,114
- Total: ~2,200

**포함 데이터**:
- ✅ Resting-state fMRI
- ✅ Structural MRI
- ✅ Phenotypic data (includes anxiety comorbidity)

**장점**:
- 큰 샘플
- Multi-site data
- 불안 comorbidity 정보

**단점**:
- ASD 중심 (불안은 부차적)
- Site 간 variability

**접근 방법**:
- Data Use Agreement 동의
- NITRC 다운로드

---

### **1.5 Enhanced Nathan Kline Institute - Rockland Sample (NKI-RS)**

**URL**: http://fcon_1000.projects.nitrc.org/indi/enhanced/

**데이터 규모**:
- N = 1,000+ (lifespan: 6-85세)

**포함 데이터**:
- ✅ Resting-state fMRI (eyes open + closed)
- ✅ Multiple scan sessions
- ✅ Comprehensive behavioral battery

**장점**:
- 고품질 multiband fMRI
- Eyes open/closed 모두 있음
- 다양한 연령대

**단점**:
- STAI 직접 측정치 확인 필요

**접근 방법**:
- 무료 다운로드 (NITRC)

---

### **1.6 Adolescent Brain Cognitive Development (ABCD) Study**

**URL**: https://abcdstudy.org/

**데이터 규모**:
- N = ~12,000 (9-10세 시작, 종단)

**포함 데이터**:
- ✅ Resting-state fMRI
- ✅ Mental health assessments (CBCL, including anxiety)
- ✅ Longitudinal (10년 추적)

**장점**:
- 매우 큰 샘플
- 종단 데이터
- 불안 관련 상세 평가

**단점**:
- 청소년 중심 (성인 아님)
- 데이터 접근 신청 필요
- 매우 큰 데이터 크기

**접근 방법**:
- NDA (National Data Archive) 신청

---

## 2. 데이터셋 선택 기준

### **우선순위 1: OpenNeuro ds002748** (Social Anxiety)
**이유**:
- ✅ 불안 직접 측정 (STAI)
- ✅ Resting-state fMRI
- ✅ 즉시 다운로드 가능
- ✅ BIDS 형식
- ⚠️ N=70 (샘플 작음)

**추천**: **파일럿 연구 및 proof-of-concept**

---

### **우선순위 2: HCP** (1200 Subjects)
**이유**:
- ✅ 고품질 resting-state fMRI
- ✅ 대규모 샘플
- ✅ 잘 정리된 데이터
- ⚠️ 불안 직접 측정 없음 → 대체 지표 사용

**전략**:
- NEO-FFI Neuroticism (불안과 높은 상관)
- DSM-oriented scales
- Proxy measures

**추천**: **메인 분석 (대규모)**

---

### **우선순위 3: UK Biobank**
**이유**:
- ✅ 가장 큰 샘플
- ✅ 불안/우울 상세 평가
- ✅ 종단 데이터
- ⚠️ 접근 승인 필요

**추천**: **확장 연구 (승인 시)**

---

## 3. 다운로드 및 전처리 계획

### **Phase 1: Pilot (ds002748)**

**다운로드**:
```bash
# DataLad 설치
pip install datalad

# 데이터셋 클론
datalad clone https://github.com/OpenNeuroDatasets/ds002748.git

# 필요한 파일만 다운로드
cd ds002748
datalad get sub-*/anat/*T1w.nii.gz
datalad get sub-*/func/*rest*bold.nii.gz
```

**예상 크기**: ~50 GB

---

### **Phase 2: Main Analysis (HCP)**

**다운로드**:
```bash
# AWS CLI 설정
aws configure

# S3에서 다운로드 (resting-state만)
aws s3 sync s3://hcp-openaccess/HCP_1200/ \
    ./HCP_1200/ \
    --exclude "*" \
    --include "*/MNINonLinear/Results/rfMRI_REST*/*" \
    --include "*/T1w/*" \
    --include "*/release-notes/*"
```

**예상 크기**: ~500 GB (resting-state only)

**대안**: Connectome Workbench 사용

---

### **전처리**

**도구**: fMRIPrep (BIDS-compatible)

```bash
# Docker로 실행
docker run -ti --rm \
    -v /path/to/data:/data:ro \
    -v /path/to/output:/out \
    nipreps/fmriprep:latest \
    /data /out participant \
    --participant-label sub-001 \
    --fs-license-file /data/license.txt
```

---

## 4. 불안 측정치 매핑

### **직접 측정 있는 경우** (ds002748)
- STAI-T (Trait Anxiety)
- STAI-S (State Anxiety)
- 직접 사용

### **대체 지표 사용** (HCP)

| 측정치 | 설명 | 불안과의 관계 |
|--------|------|--------------|
| NEO-FFI Neuroticism | 신경증 | r ≈ 0.6-0.7 with anxiety |
| DSM Anxious-Misery | DSM-oriented anxiety | 직접적 불안 지표 |
| Perceived Stress Scale | 지각된 스트레스 | 불안과 관련 |
| Life Satisfaction | 삶의 만족도 | 부적 상관 |

**전략**:
```python
# Composite anxiety score
anxiety_proxy = (
    0.4 * neuroticism +
    0.4 * dsm_anxiety +
    0.2 * perceived_stress -
    0.2 * life_satisfaction
)
```

---

## 5. 데이터 사용 계획

### **Step 1: 탐색적 분석 (ds002748, N=70)**
- SwiFT 파이프라인 구축
- Event boundary detection 검증
- Baseline 비교
- **예상 기간**: 1-2개월

### **Step 2: 메인 분석 (HCP subset, N=200-400)**
- 대규모 검증
- 불안 proxy 사용
- Robust findings
- **예상 기간**: 2-3개월

### **Step 3: 확장 (전체 HCP or UK Biobank)**
- 최대 샘플로 검증
- 개인차 탐색
- 임상 적용 가능성
- **예상 기간**: 3-4개월

---

## 6. 데이터 인용

### **ds002748**
```
Somerville, L.H., et al. (2018). Social Anxiety fMRI Dataset.
OpenNeuro. https://doi.org/10.18112/openneuro.ds002748.v1.0.0
```

### **HCP**
```
Van Essen, D.C., et al. (2013). The WU-Minn Human Connectome Project:
An overview. NeuroImage, 80, 62-79.
```

### **UK Biobank**
```
Miller, K.L., et al. (2016). Multimodal population brain imaging in
the UK Biobank prospective epidemiological study. Nature Neuroscience, 19, 1523-1536.
```

---

## 7. 윤리 및 데이터 사용 동의

### **OpenNeuro**
- Creative Commons CC0 license
- 제한 없음 (인용만 필요)

### **HCP**
- Open Access Data Use Terms 동의 필요
- 비상업적 연구 목적

### **UK Biobank**
- Application 승인 필요
- 연구 프로포절 제출
- 데이터 사용 제한 (승인된 연구만)

---

## 8. 참고 자료

- OpenNeuro: https://openneuro.org/
- HCP Wiki: https://wiki.humanconnectome.org/
- NITRC: https://www.nitrc.org/
- BIDS: https://bids.neuroimaging.io/

---

**문서 버전**: 1.0
**최종 수정**: 2024-10-23
**다음 업데이트**: 데이터셋 선택 후
