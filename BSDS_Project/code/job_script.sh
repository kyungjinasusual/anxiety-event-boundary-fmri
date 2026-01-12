#!/bin/bash
#SBATCH --job-name=EmoFilm_Extract   # 작업 이름
#SBATCH --nodes=1                    # 노드 개수
#SBATCH --ntasks=1                   # 태스크 개수
#SBATCH --cpus-per-task=4            # 코어 개수 (ROI 추출은 4개면 충분)
#SBATCH --time=01:00:00              # 최대 실행 시간 (1시간)
#SBATCH --output=./logs/log_%j.out   # 로그 저장할 곳 (%j는 Job ID)
#SBATCH --error=./logs/log_%j.err    # 에러 로그

# 1. Conda 환경 활성화 (환경 이름이 bsds라면)
source /home/castella/miniconda3_new/etc/profile.d/conda.sh
conda activate bsds

# 2. 파이썬 실행 ($1은 우리가 밖에서 던져줄 피험자 ID)
echo "Processing Subject: $1"
python run_extraction.py $1