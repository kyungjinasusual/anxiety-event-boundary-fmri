# 데이터 확인 스크립트: 출석부 생성
# 작성일: 2025-12-11

import os
import glob
import pandas as pd
import sys

# [수정됨] Emo-Film -> Emo-FilM (대문자 M 주의!)
BASE_DIR = '/storage/bigdata/Emo-FilM/brain_data/derivatives/preprocessing'

# 혹시 경로 자체가 틀렸는지 먼저 확인
if not os.path.exists(BASE_DIR):
    print(f"❌ 오류: 기본 경로를 찾을 수 없습니다: {BASE_DIR}")
    print("경로 철자(대소문자)를 다시 확인해주세요!")
    sys.exit()

# S01 ~ S32 (필요시 숫자 조정)
SUBJECTS = [f"sub-S{i:02d}" for i in range(1, 33)] 
TASKS = ['BigBuckBunny', 'FirstBite', 'YouAgain', 'Rest']

data_status = []

print(f"📂 검색 경로: {BASE_DIR} (확인됨)")

for sub in SUBJECTS:
    row = {'Subject': sub}
    for task in TASKS:
        # 파일명 패턴
        file_pattern = os.path.join(
            BASE_DIR, sub, 'ses-1', 'func',
            f"{sub}_ses-1_task-{task}_space-MNI_desc-ppres_bold.nii.gz"
        )
        
        files = glob.glob(file_pattern)
        if len(files) > 0:
            row[task] = "O"
        else:
            row[task] = "X"
            # 디버깅: 첫 번째로 못 찾은 파일의 경로를 출력해서 확인해봄
            if sub == 'sub-S01' and task == 'BigBuckBunny':
                 print(f"🔍 [디버그] S01 파일을 못 찾았습니다. 검색한 경로:\n   {file_pattern}")

    data_status.append(row)

df = pd.DataFrame(data_status)
print("\n========== 데이터 출석부 (재시도) ==========")
print(df)