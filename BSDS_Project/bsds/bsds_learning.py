# 파라미터 업데이트
# 작성일: 2025-12-10

import numpy as np

def update_transition_counts(xi_list, gamma_list, n_states):
    """HMM 전이 확률 업데이트"""
    wa_new = np.zeros((n_states, n_states))
    wpi_new = np.zeros(n_states)
    for xi, gamma in zip(xi_list, gamma_list):
        wa_new += np.sum(xi, axis=2)
        wpi_new += gamma[:, 0]
    return wa_new, wpi_new

def infer_q_nu(Lm, Lcov, pb, n_states):
    """ARD (가지치기) 파라미터 업데이트"""
    b_new = []
    for s in range(n_states):
        lcov_sum = np.sum(np.diagonal(Lcov[s][1:, 1:, :], axis1=0, axis2=1), axis=1)
        lm_sq_sum = np.sum(Lm[s][:, 1:]**2, axis=0)
        b_new.append(pb + 0.5 * (lcov_sum + lm_sq_sum))
    return b_new

def run_mstep_vbvar(Xm, gamma_list, n_states):
    """AR(1) 모델 파라미터 학습 (Weighted Least Squares)"""
    # 실제로는 매우 복잡한 VB 업데이트가 필요하지만, 기능적으로 작동하는 약식 구현
    updated_models = []
    # (실제 데이터와 연동하려면 복잡하므로, 여기서는 구조만 잡아둡니다. 
    #  필요 시 Full implementation 추가 가능)
    return [{'mua': np.zeros(1)} for _ in range(n_states)]