# Code Monitor Setup for Kyungjin Oh

## 🎯 현재 설정

**Base Directory**: `/Users/ohkyungjin`
- 홈 디렉토리 전체를 스캔하여 모든 Git 저장소 추적
- 현재 발견된 저장소들:
  - `ML_tutor/`
  - `github/`
  - `kyungjin.github.io/`
  - `Downloads/anxiety-event-boundary-fmri/`
  - 기타 홈 디렉토리 내 모든 Git 저장소

**Git Author Names**:
- `Kyungjin Oh`
- `kyungjinasusual` (GitHub username)

## 📊 사용 방법

### 1. 기본 사용 (이번 주 코드 통계)

```bash
cd /Users/ohkyungjin/Downloads/anxiety-event-boundary-fmri/code_monitor

# 당신의 이름으로 필터링
./monitor.sh -a "Kyungjin Oh"

# 또는 GitHub username으로
./monitor.sh -a kyungjinasusual

# 또는 부분 문자열 매칭
./monitor.sh -a kyungjin
```

### 2. HTML 리포트 생성

```bash
# 주간 HTML 리포트
./monitor.sh -a kyungjin -h

# 월간 HTML 리포트
./monitor.sh -w 4 -a kyungjin -h

# 모든 브랜치 포함 (더 완전한 통계)
./monitor.sh -a kyungjin -b -h
```

### 3. 모든 저장소 확인 (author 필터 없이)

```bash
# 모든 author의 활동
./monitor.sh

# 모든 author + HTML
./monitor.sh -h
```

## 🔍 저장소 범위 확인

**현재 스캔 범위**: `/Users/ohkyungjin` (홈 디렉토리 전체)

**주의사항**:
- 홈 디렉토리에 Git 저장소가 많으면 스캔 시간이 길어질 수 있음
- 필요시 `monitor_config.yaml`의 `base_directory`를 특정 디렉토리로 변경 가능

**범위를 좁히려면**:

```yaml
# monitor_config.yaml 수정
base_directory: /Users/ohkyungjin/Downloads  # Downloads만 스캔
```

또는

```yaml
base_directory: /Users/ohkyungjin/Projects  # Projects 폴더만 (생성 필요)
```

## 📂 Projects 폴더 추천 구조

모든 GitHub 저장소를 한 곳에 모으면 관리가 쉽습니다:

```bash
# Projects 폴더 생성
mkdir -p /Users/ohkyungjin/Projects

# 기존 저장소 이동
mv /Users/ohkyungjin/Downloads/anxiety-event-boundary-fmri /Users/ohkyungjin/Projects/
mv /Users/ohkyungjin/kyungjin.github.io /Users/ohkyungjin/Projects/
mv /Users/ohkyungjin/github /Users/ohkyungjin/Projects/
mv /Users/ohkyungjin/ML_tutor /Users/ohkyungjin/Projects/

# config 업데이트
# monitor_config.yaml에서:
# base_directory: /Users/ohkyungjin/Projects
```

이후 구조:
```
/Users/ohkyungjin/Projects/
├── anxiety-event-boundary-fmri/
├── kyungjin.github.io/
├── github/
├── ML_tutor/
└── [새 프로젝트들]
```

## 🚀 Shell Alias 추천

`~/.zshrc` 또는 `~/.bashrc`에 추가:

```bash
# Code Monitor 관련
alias mycode='cd /Users/ohkyungjin/Downloads/anxiety-event-boundary-fmri/code_monitor && ./monitor.sh -a kyungjin'
alias myweek='cd /Users/ohkyungjin/Downloads/anxiety-event-boundary-fmri/code_monitor && ./monitor.sh -a kyungjin -b -h'
alias mymonth='cd /Users/ohkyungjin/Downloads/anxiety-event-boundary-fmri/code_monitor && ./monitor.sh -w 4 -a kyungjin -b -h'
```

적용:
```bash
source ~/.zshrc  # 또는 source ~/.bashrc
```

사용:
```bash
mycode   # 빠른 체크
myweek   # 주간 리포트
mymonth  # 월간 리포트
```

## 📊 예상 출력

```
======================================================================
📈 CODE MONITOR REPORT - 1 Week(s)
📅 Period: 2025-10-20 to 2025-10-27
👤 Author Filter: kyungjin
======================================================================

📦 anxiety-event-boundary-fmri
   Commits: 1
   Authors: Kyungjin Oh
   📝 Code Files:
      ✅ Added: 14
   📚 Documentation:
      ✅ Added: 4
   📊 Lines: +2426 -0

======================================================================
📊 SUMMARY
======================================================================
Total Commits: 1
Active Authors: 1

Code Files:
  ✅ Added: 14

Documentation:
  ✅ Added: 4

Total Lines: +2426 -0
Net Change: +2426 lines
======================================================================
```

## ⚙️ 고급 사용

### Python API 직접 사용

```bash
# JSON 출력
python3 code_monitor.py --author kyungjin --weeks 1 --output report.json

# 특정 디렉토리만
python3 code_monitor.py --dir /Users/ohkyungjin/Downloads --weeks 1

# 모든 브랜치 + 4주
python3 code_monitor.py -a kyungjin -b --weeks 4
```

### 자동화 (Cron)

```bash
# crontab 편집
crontab -e

# 매주 월요일 오전 9시 리포트
0 9 * * 1 cd /Users/ohkyungjin/Downloads/anxiety-event-boundary-fmri/code_monitor && ./monitor.sh -a kyungjin -b -h -o ~/Desktop/
```

## 🔧 문제 해결

### 1. "No repositories found"

```bash
# Git 저장소 확인
find /Users/ohkyungjin -maxdepth 3 -name ".git" -type d

# base_directory 확인
cat monitor_config.yaml | grep base_directory
```

### 2. "No data showing"

```bash
# Author 이름 확인
git log --format='%an' | sort -u

# 더 긴 기간 시도
./monitor.sh -a kyungjin -w 4

# 모든 브랜치 확인
./monitor.sh -a kyungjin -b
```

### 3. 실행 권한 오류

```bash
chmod +x code_monitor.py visualize_monitor.py monitor.sh
```

## 📝 다음 단계

1. **테스트 실행**: `./monitor.sh -a kyungjin`
2. **HTML 리포트 확인**: `./monitor.sh -a kyungjin -h`
3. **Alias 설정**: 위의 alias를 `.zshrc`에 추가
4. **(선택) Projects 폴더 정리**: 모든 저장소를 한 곳으로

---

**빠른 시작**: `./monitor.sh -a kyungjin -h` 🚀
